// Fill out your copyright notice in the Description page of Project Settings.


#include "DQNComponent.h"
#include "TcpServer.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"

namespace
{
	static float CM_TO_M = 1.0f / 100.0f;
}
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
		FVector2D d;
		FVector2D v;
		TArray<float> o; ComputeObs(o, d, v);
		//PrevDist = d;
	}
}

void UDQNComponent::ComputeObs(TArray<float>& OutObs, FVector2D& OutRelativeDistance, FVector2D& OutRelativeVelocity) const
{
	OutObs.Reset();
	OutObs.SetNum(4);

	const FVector PlayerPosition = PlayerActor ? PlayerActor->GetActorLocation() : FVector::ZeroVector;
	const FVector AgentPosition = GetOwner()->GetActorLocation();

	const FVector RelativeDistance = (PlayerPosition - AgentPosition) * CM_TO_M;
	
	FVector PlayerVelocity = FVector::Zero();

	if (PlayerActor)
	{
		PlayerVelocity = PlayerActor->GetVelocity();
	}

	FVector AgentVelocity = FVector::ZeroVector;
	UPrimitiveComponent* RootPrim = Cast<UPrimitiveComponent>(GetOwner()->GetRootComponent());
	if (RootPrim)
		AgentVelocity = RootPrim->GetPhysicsLinearVelocity();

	auto RelativeVelocity = (AgentVelocity - PlayerVelocity) * CM_TO_M;

	// Normalize for the network DQN
	auto RelativeVelocityNormalized = RelativeVelocity / MaxRelativeSpeed;
	auto RelativeDistanceNormalized = RelativeDistance / MaxDistance;

	OutObs[0] = FMath::Clamp(RelativeDistanceNormalized.X, -1.f, 1.f);
	OutObs[1] = FMath::Clamp(RelativeDistanceNormalized.Y, -1.f, 1.f);
	OutObs[2] = FMath::Clamp(RelativeVelocityNormalized.X, -1.f, 1.f);
	OutObs[3] = FMath::Clamp(RelativeVelocityNormalized.Y, -1.f, 1.f);

	OutRelativeDistance = FVector2D(RelativeDistance.X, RelativeDistance.Y);
	OutRelativeVelocity = FVector2D(RelativeVelocity.X, RelativeVelocity.Y);
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

	FVector2D Dist;
	FVector2D Speed;
	TArray<float> Obs;
	ComputeObs(Obs, Dist, Speed);

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
	FVector2D RelDist;
	FVector2D RelVelocity;
	TArray<float> Obs;
	ComputeObs(Obs, RelDist, RelVelocity);
		
	float Reward = 0.0f;

	auto RelVelocityClosing = RelDist.GetSafeNormal().Dot(RelVelocity) / MaxRelativeSpeed;
	
	//UE_LOG(LogTemp, Warning, TEXT("Velocite %f"), RelVelocityClosing);

	Reward += RelVelocityClosing;

	auto RelDistanceClosing = (1 - RelDist.Length() / MaxDistance);
	//UE_LOG(LogTemp, Warning, TEXT("Distance %f"), RelDistanceClosing);

	Reward += RelDistanceClosing;

	StepCount++;
	GlobalT++;

	const bool bTimeout = (StepCount >= MaxStepsPerEpisode);
	const bool bTooFar = (RelDist.Length() >= MaxDistance);
	
	bool bDone = bTimeout || bTooFar;

	if (bTooFar) Reward -= 0.5f;
	
	//UE_LOG(LogTemp, Warning, TEXT("Reward %f"), Reward);

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
		DrawDebugSphere(GetWorld(), PlayerActor->GetActorLocation(), MaxDistance / CM_TO_M, 16, FColor::Green, false, 0.1f, 0, 2.f);
	}
}

