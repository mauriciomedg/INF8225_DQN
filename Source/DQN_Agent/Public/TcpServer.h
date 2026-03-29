// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "HAL/ThreadSafeBool.h"
#include "TcpServer.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FTcpServerLineEvent, const FString&, Line);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FTcpServerErrorEvent, const FString&, Error);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FTcpServerClientEvent);
/**
 * 
 */

UCLASS()
class DQN_AGENT_API UTcpServer : public UObject
{
	GENERATED_BODY()

public:

	// Port to listen on (0.0.0.0:Port)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TCP Server")
	int32 Port = 7777;

	// Start listening (creates ListenSocket, binds, listens)
	UFUNCTION(BlueprintCallable, Category = "TCP Server")
	bool StartServer();

	// Stop everything (drops client if connected, closes listen socket)
	UFUNCTION(BlueprintCallable, Category = "TCP Server")
	void StopServer();

	// Send a newline-delimited JSON line to the connected client.
	// The string passed should be valid JSON without trailing newline; we append '\n'.
	UFUNCTION(BlueprintCallable, Category = "TCP Server")
	bool SendJsonLine(const FString& JsonLine);

	// Is there a connected client?
	UFUNCTION(BlueprintPure, Category = "TCP Server")
	bool IsClientConnected() const;

	// Events (GameThread)
	UPROPERTY(BlueprintAssignable, Category = "TCP Server")
	FTcpServerLineEvent OnJsonLine;

	UPROPERTY(BlueprintAssignable, Category = "TCP Server")
	FTcpServerErrorEvent OnStreamError;

	UPROPERTY(BlueprintAssignable, Category = "TCP Server")
	FTcpServerClientEvent OnClientConnected;

	UPROPERTY(BlueprintAssignable, Category = "TCP Server")
	FTcpServerClientEvent OnClientDisconnected;


public:
	UTcpServer();

	void BeginPlay();
	void EndPlay();
	void Tick(float DeltaTime);

	// Called by worker on GameThread (do not call directly)
	void HandleLine_GameThread(const FString& Line);
	void HandleError_GameThread(const FString& Err);
	void HandleDone_GameThread();

private:
	// Accept connection if pending (called from Tick)
	void PollAccept();

	// Drop current client connection (keeps server listening)
	void StopClient();

private:
	// Listening socket (0.0.0.0:Port)
	class FSocket* ListenSocket = nullptr;

	// Connected client socket (returned by Accept)
	class FSocket* ClientSocket = nullptr;

	// Reader worker + thread
	class FRunnable* Worker = nullptr;
	class FRunnableThread* WorkerThread = nullptr;

};
