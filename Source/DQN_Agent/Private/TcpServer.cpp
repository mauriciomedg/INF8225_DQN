// Fill out your copyright notice in the Description page of Project Settings.


#include "TcpServer.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "IPAddress.h"

#include "Async/Async.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "HAL/ThreadSafeBool.h"

namespace
{
	// Reads bytes from socket, emits complete lines (newline-delimited).
	// IMPORTANT: Socket pointer must remain valid until the worker exits.
	class FJsonLineReaderWorker : public FRunnable
	{
	public:
		FJsonLineReaderWorker(UTcpServer* InOwner, FSocket* InSocket)
			: Owner(InOwner)
			, Socket(InSocket)
		{
		}

		virtual uint32 Run() override
		{
			if (!Socket || !Owner)
				return 0;

			TArray<uint8> Pending;
			Pending.Reserve(4096);

			uint8 Temp[1024];

			while (!bStopRequested)
			{
				int32 BytesRead = 0;
				const bool bOk = Socket->Recv(Temp, sizeof(Temp), BytesRead);

				// If we're stopping, Recv failures/zero bytes are expected (socket shutdown/close).
				if (bStopRequested)
				{
					DispatchDone();
					return 0;
				}

				if (!bOk)
				{
					DispatchError(TEXT("recv failed"));
					return 0;
				}

				if (BytesRead <= 0)
				{
					DispatchError(TEXT("connection closed"));
					return 0;
				}

				Pending.Append(Temp, BytesRead);

				// Extract complete lines
				while (true)
				{
					int32 NewlineIndex = INDEX_NONE;
					for (int32 i = 0; i < Pending.Num(); ++i)
					{
						if (Pending[i] == '\n')
						{
							NewlineIndex = i;
							break;
						}
					}

					if (NewlineIndex == INDEX_NONE)
						break;

					// Line bytes [0..NewlineIndex-1]
					TArray<uint8> LineBytes;
					LineBytes.Append(Pending.GetData(), NewlineIndex);

					// Remove consumed bytes (+ '\n')
					Pending.RemoveAt(0, NewlineIndex + 1, /*bAllowShrinking*/ false);

					// Trim optional '\r' (CRLF)
					if (LineBytes.Num() > 0 && LineBytes.Last() == '\r')
						LineBytes.Pop(false);

					// Null-terminate
					LineBytes.Add(0);

					const char* Utf8 = reinterpret_cast<const char*>(LineBytes.GetData());
					const FString Line = UTF8_TO_TCHAR(Utf8);

					DispatchLine(Line);
				}
			}

			DispatchDone();
			return 0;
		}

		virtual void Stop() override
		{
			bStopRequested = true;
		}

	private:
		UTcpServer* Owner = nullptr;
		FSocket* Socket = nullptr;
		FThreadSafeBool bStopRequested = false;

		void DispatchLine(const FString& Line)
		{
			if (!Owner) return;
			AsyncTask(ENamedThreads::GameThread, [Owner = Owner, Line]()
				{
					if (IsValid(Owner))
					{
						Owner->HandleLine_GameThread(Line);
					}
				});
		}

		void DispatchError(const FString& Err)
		{
			if (!Owner) return;
			AsyncTask(ENamedThreads::GameThread, [Owner = Owner, Err]()
				{
					if (IsValid(Owner))
					{
						Owner->HandleError_GameThread(Err);
					}
				});
		}

		void DispatchDone()
		{
			if (!Owner) return;
			AsyncTask(ENamedThreads::GameThread, [Owner = Owner]()
				{
					if (IsValid(Owner))
					{
						Owner->HandleDone_GameThread();
					}
				});
		}
	};

	static bool SendAll(FSocket* Socket, const uint8* Data, int32 DataSize)
	{
		if (!Socket || !Data || DataSize <= 0) return false;

		int32 TotalSent = 0;
		while (TotalSent < DataSize)
		{
			int32 SentNow = 0;
			if (!Socket->Send(Data + TotalSent, DataSize - TotalSent, SentNow) || SentNow <= 0)
			{
				return false;
			}
			TotalSent += SentNow;
		}
		return true;
	}
}

UTcpServer::UTcpServer()
{
}

void UTcpServer::BeginPlay()
{
	// Optional: auto-start. If you prefer manual, remove these two lines.
	StartServer();
}

void UTcpServer::EndPlay()
{
	StopServer();
}

void UTcpServer::Tick(float DeltaTime)
{
	PollAccept();
}

bool UTcpServer::StartServer()
{
	StopServer(); // cleanup if previously running

	ISocketSubsystem* Subsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
	if (!Subsystem)
	{
		UE_LOG(LogTemp, Error, TEXT("TCP Server: no socket subsystem"));
		return false;
	}

	ListenSocket = Subsystem->CreateSocket(NAME_Stream, TEXT("TcpListen"), false);
	if (!ListenSocket)
	{
		UE_LOG(LogTemp, Error, TEXT("TCP Server: failed to create listen socket"));
		return false;
	}

	ListenSocket->SetReuseAddr(true);
	ListenSocket->SetNoDelay(true);
	ListenSocket->SetNonBlocking(true); // Accept without blocking game thread

	// Bind to 0.0.0.0:Port (all interfaces)
	TSharedRef<FInternetAddr> Addr = Subsystem->CreateInternetAddr();
	Addr->SetAnyAddress();
	Addr->SetPort(Port);

	if (!ListenSocket->Bind(*Addr))
	{
		UE_LOG(LogTemp, Error, TEXT("TCP Server: bind failed on port %d"), Port);
		Subsystem->DestroySocket(ListenSocket);
		ListenSocket = nullptr;
		return false;
	}

	const int32 Backlog = 1;
	if (!ListenSocket->Listen(Backlog))
	{
		UE_LOG(LogTemp, Error, TEXT("TCP Server: listen failed"));
		Subsystem->DestroySocket(ListenSocket);
		ListenSocket = nullptr;
		return false;
	}

	UE_LOG(LogTemp, Log, TEXT("TCP Server: listening on port %d"), Port);
	return true;
}

void UTcpServer::PollAccept()
{
	if (!ListenSocket || ClientSocket) return; // already have a client

	bool bHasPending = false;
	if (!ListenSocket->HasPendingConnection(bHasPending) || !bHasPending)
		return;

	ISocketSubsystem* Subsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
	TSharedRef<FInternetAddr> ClientAddr = Subsystem->CreateInternetAddr();

	ClientSocket = ListenSocket->Accept(*ClientAddr, TEXT("TcpClientConn"));
	if (!ClientSocket)
	{
		UE_LOG(LogTemp, Warning, TEXT("TCP Server: Accept() returned null"));
		return;
	}

	ClientSocket->SetNonBlocking(false); // blocking recv is fine in worker thread
	ClientSocket->SetNoDelay(true);

	UE_LOG(LogTemp, Log, TEXT("TCP Server: client connected"));
	OnClientConnected.Broadcast();

	// Start reader thread on ClientSocket
	Worker = new FJsonLineReaderWorker(this, ClientSocket);
	WorkerThread = FRunnableThread::Create(Worker, TEXT("TcpServer_Reader"));

	if (!WorkerThread)
	{
		UE_LOG(LogTemp, Error, TEXT("TCP Server: failed to create reader thread"));
		delete Worker; Worker = nullptr;

		ClientSocket->Close();
		Subsystem->DestroySocket(ClientSocket);
		ClientSocket = nullptr;
		OnClientDisconnected.Broadcast();
		return;
	}

	UE_LOG(LogTemp, Log, TEXT("TCP Server: stream started"));
}

bool UTcpServer::IsClientConnected() const
{
	return ClientSocket && ClientSocket->GetConnectionState() == ESocketConnectionState::SCS_Connected;
}

bool UTcpServer::SendJsonLine(const FString& JsonLine)
{
	if (!IsClientConnected())
		return false;

	const FString LineToSend = JsonLine + TEXT("\n");
	FTCHARToUTF8 Converter(*LineToSend);
	const int32 DataSize = Converter.Length();

	if (!SendAll(ClientSocket, reinterpret_cast<const uint8*>(Converter.Get()), DataSize))
	{
		UE_LOG(LogTemp, Error, TEXT("TCP Server: send failed (dropping client)"));
		StopClient();
		return false;
	}
	return true;
}

void UTcpServer::StopClient()
{
	// Stop worker
	if (Worker) Worker->Stop();

	// Unblock Recv() safely
	if (ClientSocket)
	{
		ClientSocket->Shutdown(ESocketShutdownMode::ReadWrite);
		ClientSocket->Close();
	}

	// Join worker thread while socket pointer still valid
	if (WorkerThread)
	{
		WorkerThread->WaitForCompletion();
		delete WorkerThread;
		WorkerThread = nullptr;
	}

	if (Worker)
	{
		delete Worker;
		Worker = nullptr;
	}

	// Destroy client socket
	if (ClientSocket)
	{
		ISocketSubsystem* Subsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
		if (Subsystem) Subsystem->DestroySocket(ClientSocket);
		ClientSocket = nullptr;
	}

	OnClientDisconnected.Broadcast();
}

void UTcpServer::StopServer()
{
	StopClient();

	if (ListenSocket)
	{
		ListenSocket->Close();
		ISocketSubsystem* Subsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
		if (Subsystem) Subsystem->DestroySocket(ListenSocket);
		ListenSocket = nullptr;
	}
}

void UTcpServer::HandleLine_GameThread(const FString& Line)
{
	OnJsonLine.Broadcast(Line);
}

void UTcpServer::HandleError_GameThread(const FString& Err)
{
	OnStreamError.Broadcast(Err);

	// Drop this client only; keep listening so Python can reconnect.
	StopClient();
}

void UTcpServer::HandleDone_GameThread()
{
	// Not used for RL; kept for compatibility.
	// You can hook this if you want, but typically "done" is a field in JSON messages.
}