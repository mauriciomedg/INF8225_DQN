// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "DQNComponent.generated.h"

class UTcpServer;

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class DQN_AGENT_API UDQNComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UDQNComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Networking")
	TObjectPtr<UTcpServer> TcpServer;

	UPROPERTY(EditAnywhere)
	TObjectPtr<AActor> PlayerActor;

	UPROPERTY(EditAnywhere)
	FVector TargetLocation;

	UPROPERTY(EditAnywhere)
	float StepHz = 20.f;

	UPROPERTY(EditAnywhere)
	float ImpulseStrength = 600.f;

	UPROPERTY(EditAnywhere)
	float CatchRadius = 20.f;

	UPROPERTY(EditAnywhere)
	float MaxDistance = 3000.f;

	UPROPERTY(EditAnywhere)
	int32 MaxStepsPerEpisode = 300;

private:
	FTimerHandle StepTimer;

	// latest action received from python
	bool bHasPendingAction = false;
	int32 PendingAction = 0;

	bool bWaitingForPostPhysics = false;

	int32 StepCount = 0;
	int32 GlobalT = 0;
	float PrevDist = 0.f;

private:
	UFUNCTION()
	void OnTcpLine(const FString& Line);

	UFUNCTION()
	void OnClientConnected();

	void StepLoop();

	void ResetEpisode();

	void ApplyAction(int32 A);

	void SendReset();
	void SendStep(const TArray<float>& Obs, float Reward, bool bDone);

	void ComputeObs(TArray<float>& OutObs, float& OutDist) const;
		
};
