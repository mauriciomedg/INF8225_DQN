// Fill out your copyright notice in the Description page of Project Settings.


#include "DQNComponent.h"
#include "TcpServer.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"

// Sets default values for this component's properties
UDQNComponent::UDQNComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.TickGroup = TG_PostPhysics;

	TcpServer = CreateDefaultSubobject<UTcpServer>(TEXT("UTcpServer"));

	// ...
}


// Called when the game starts
void UDQNComponent::BeginPlay()
{
	Super::BeginPlay();

	if (TcpServer)
		TcpServer->BeginPlay();
	
	check(TcpServer);

	// bind to incoming lines
	TcpServer->OnJsonLine.AddDynamic(this, &UDQNComponent::OnTcpLine);
	TcpServer->OnClientConnected.AddDynamic(this, &UDQNComponent::OnClientConnected);

	if (!PlayerActor)
	{
		APlayerController* PC = GetWorld()->GetFirstPlayerController();
		if (PC)
		{
			PlayerActor = Cast<AActor>(PC->GetPawn());
		}
	}
		
	const float Dt = 1.f / FMath::Max(1.f, StepHz);
	GetOwner()->GetWorldTimerManager().SetTimer(StepTimer, this, &UDQNComponent::StepLoop, Dt, true);
		
}

void UDQNComponent::OnClientConnected()
{
	UE_LOG(LogTemp, Log, TEXT("DQN: client connected, starting episode"));

	ResetEpisode();
	SendReset();
}


void UDQNComponent::OnTcpLine(const FString& Line)
{
	// Expect {"type":"action","a":int}
	TSharedPtr<FJsonObject> Root;
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Line);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
		return;

	FString Type;
	if (!Root->TryGetStringField(TEXT("type"), Type))
		return;

	if (Type == TEXT("action"))
	{
		int32 A = 0;
		if (Root->TryGetNumberField(TEXT("a"), A))
		{
			PendingAction = A;
			bHasPendingAction = true;
		}
	}
}

void UDQNComponent::StepLoop()
{
	if (!TcpServer || !TcpServer->IsClientConnected())
		return;

	// Need an action from python
	if (!bHasPendingAction)
		return;

	// If we already applied an action and haven't sent the resulting step yet, wait.
	if (bWaitingForPostPhysics)
		return;

	/// Capture s_t BEFORE applying action (optional but recommended for debug)
	//float Dist = 0.f;
	//ComputeObs(LastObs, Dist);

	ApplyAction(PendingAction);
	bHasPendingAction = false;

	// Now we wait until PostPhysics Tick to measure s_{t+1} and compute reward
	bWaitingForPostPhysics = true;

}

static UPrimitiveComponent* FindSimulatingPrim(AActor* Owner)
{
	if (!Owner) return nullptr;

	// Prefer root if it simulates
	if (UPrimitiveComponent* RootPrim = Cast<UPrimitiveComponent>(Owner->GetRootComponent()))
	{
		if (RootPrim->IsSimulatingPhysics())
			return RootPrim;
	}

	// Otherwise find any simulating primitive
	TArray<UPrimitiveComponent*> Prims;
	Owner->GetComponents<UPrimitiveComponent>(Prims);
	for (UPrimitiveComponent* P : Prims)
	{
		if (P && P->IsSimulatingPhysics())
			return P;
	}
	return nullptr;
}

void UDQNComponent::ResetEpisode()
{
	StepCount = 0;

	// Reset Agent location
	UPrimitiveComponent* Prim = FindSimulatingPrim(GetOwner());
	if (!Prim) return;

	if (FBodyInstance* BI = Prim->GetBodyInstance())
	{
		BI->SetLinearVelocity(FVector::ZeroVector, false);
		BI->SetAngularVelocityInRadians(FVector::ZeroVector, false);

		// Teleport physics body transform directly
		BI->SetBodyTransform(FTransform(FRotator::ZeroRotator, AgentStartLocation), ETeleportType::TeleportPhysics);

		// Optional: ensure it's awake
		BI->WakeInstance();
	}

	// reset player	
	ACharacter* Character = Cast<ACharacter>(PlayerActor);
	if (Character)
		Character->TeleportTo(PlayerStartLocation, FRotator::ZeroRotator);
	
	{
		float d = 0; 
		TArray<float> o; ComputeObs(o, d);
		PrevDist = d;
	}
}

void UDQNComponent::ComputeObs(TArray<float>& OutObs, float& OutDist) const
{
	OutObs.Reset();
	OutObs.SetNum(4);

	const FVector P = PlayerActor ? PlayerActor->GetActorLocation() : FVector::ZeroVector;
	const FVector S = GetOwner()->GetActorLocation();

	const FVector D = P - S;
	OutDist = FVector(D.X, D.Y, 0).Size();

	const float Inv = 1.f / FMath::Max(1.f, MaxDistance);

	FVector PlayerVelocity = FVector::Zero();

	if (PlayerActor)
		PlayerVelocity = PlayerActor->GetVelocity();

	OutObs[0] = (D.X + PlayerVelocity.X) * Inv;
	OutObs[1] = (D.Y + PlayerVelocity.Y) * Inv;

	FVector V = FVector::ZeroVector;
	UPrimitiveComponent* RootPrim = Cast<UPrimitiveComponent>(GetOwner()->GetRootComponent());
	if (RootPrim)
		V = RootPrim->GetPhysicsLinearVelocity();

	const float VNorm = 1.f / 100.f;
	OutObs[2] = FMath::Clamp(V.X * VNorm, -1.f, 1.f);
	OutObs[3] = FMath::Clamp(V.Y * VNorm, -1.f, 1.f);

	//UE_LOG(LogTemp, Warning, TEXT("V raw X %f Y %f"), float(V.X), float(V.Y));
	//UE_LOG(LogTemp, Warning, TEXT("V X %f Y %f"), OutObs[2], OutObs[3]);
}

void UDQNComponent::ApplyAction(int32 A)
{
	UPrimitiveComponent* RootPrim = Cast<UPrimitiveComponent>(GetOwner()->GetRootComponent());
	if (!RootPrim) return;

	FVector Imp(0, 0, 0);
	switch (A)
	{
	case 0: Imp = FVector(1, 0, 0); break;
	case 1: Imp = FVector(-1, 0, 0); break;
	case 2: Imp = FVector(0, 1, 0); break;
	case 3: Imp = FVector(0, -1, 0); break;
	default: break;
	}

	RootPrim->AddImpulse(Imp * ImpulseStrength, NAME_None, true);
}

void UDQNComponent::SendReset()
{
	if (!TcpServer->IsClientConnected())
		return;

	float Dist = 0.f;
	TArray<float> Obs;
	ComputeObs(Obs, Dist);

	// build JSON
	TSharedPtr<FJsonObject> Obj = MakeShared<FJsonObject>();
	Obj->SetStringField(TEXT("type"), TEXT("reset"));
	Obj->SetNumberField(TEXT("t"), GlobalT);

	TArray<TSharedPtr<FJsonValue>> ObsArr;
	for (float v : Obs) ObsArr.Add(MakeShared<FJsonValueNumber>(v));
	Obj->SetArrayField(TEXT("obs"), ObsArr);

	FString Out;
	TSharedRef<TJsonWriter<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>> Writer =
		TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Out);

	//TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Out);
	FJsonSerializer::Serialize(Obj.ToSharedRef(), Writer);

	TcpServer->SendJsonLine(Out);
}

void UDQNComponent::SendStep(const TArray<float>& Obs, float Reward, bool bDone)
{
	TSharedPtr<FJsonObject> Obj = MakeShared<FJsonObject>();
	Obj->SetStringField(TEXT("type"), TEXT("step"));
	Obj->SetNumberField(TEXT("t"), GlobalT);
	Obj->SetNumberField(TEXT("reward"), Reward);
	Obj->SetBoolField(TEXT("done"), bDone);

	TArray<TSharedPtr<FJsonValue>> ObsArr;
	for (float v : Obs) ObsArr.Add(MakeShared<FJsonValueNumber>(v));
	Obj->SetArrayField(TEXT("obs"), ObsArr);

	FString Out;
	//TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Out);
	TSharedRef<TJsonWriter<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>> Writer =
		TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Out);
	FJsonSerializer::Serialize(Obj.ToSharedRef(), Writer);

	TcpServer->SendJsonLine(Out);
}

void UDQNComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (TcpServer)
		TcpServer->EndPlay();

	Super::EndPlay(EndPlayReason);
}

// Called every frame
void UDQNComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (TcpServer)
		TcpServer->Tick(DeltaTime);

	// Only generate a transition after we applied an action
	if (!TcpServer || !TcpServer->IsClientConnected())
		return;

	if (!bWaitingForPostPhysics)
		return;

	// Now physics has advanced since the action was applied
	float Dist = 0.f;
	TArray<float> Obs;
	ComputeObs(Obs, Dist);
		
	float Reward = 0.0f;

	// progress reward
	Reward += (PrevDist - Dist) / MaxDistance;  // roughly in [-0.5,0.5]
	
	StepCount++;
	GlobalT++;

	const bool bTimeout = (StepCount >= MaxStepsPerEpisode);
	const bool bTooFar = (Dist >= MaxDistance);
	
	bool bDone = bTimeout || bTooFar;

	// “stay close” bonus
	if (Dist <= CatchRadius && (PrevDist - Dist > 0))
		Reward += 0.2f;

	if (bTooFar) Reward -= 0.5f;
	
	PrevDist = Dist;
	//UE_LOG(LogTemp, Warning, TEXT("Reward %f Dist %f"), Reward, Dist);

	// Send the resulting state after the action
	SendStep(Obs, Reward, bDone);

	// We have completed one env step
	bWaitingForPostPhysics = false;

	if (bDone)
	{
		ResetEpisode();
		SendReset();
	}

	// Debug draw ok here
	if (PlayerActor)
	{
		DrawDebugSphere(GetWorld(), PlayerActor->GetActorLocation(), CatchRadius, 16, FColor::Green, false, 0.1f, 0, 2.f);
	}
}

