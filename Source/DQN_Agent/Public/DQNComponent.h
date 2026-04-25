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

	UPROPERTY(VisibleAnywhere)
	TObjectPtr<AActor> PlayerActor;

	UPROPERTY(EditAnywhere)
	FVector PlayerStartLocation;

	UPROPERTY(EditAnywhere)
	FVector AgentStartLocation;

	UPROPERTY(EditAnywhere)
	float StepHz = 20.f;

	UPROPERTY(EditAnywhere)
	float ImpulseStrength = 600.f;

	UPROPERTY(EditAnywhere)
	float MaxDistance = 30.f; // m

	UPROPERTY(EditAnywhere, Category = "DQN")
	float MaxRelativeSpeed = 18.f; // m/s

	UPROPERTY(EditAnywhere, Category = "DQN")
	int32 MaxStepsPerEpisode = 300;

	UPROPERTY(EditAnywhere, Category = "DQN", meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float Gamma = 0.99f;																					   
	UPROPERTY(EditAnywhere, Category = "DQN")
	bool bInferenceMode = true;

private:
	FTimerHandle StepTimer;

	// latest action received from python
	bool bHasPendingAction = false;
	int32 PendingAction = 0;

	bool bWaitingForPostPhysics = false;

	int32 StepCount = 0;
	int32 GlobalT = 0;
	float PrevDist = 0.f;
	float PrevSpeed = 0.f;
	float PrevPotential = 0.f;
	bool  bHasPrevPotential = false;
 
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

	void ComputeObs(TArray<float>& OutObs, FVector2D& OutRelativeDistance, FVector2D& OutRelativeVelocity) const;
    float ComputePotential(const FVector2D& RelDist) const;														
		
	void TickForTraining();
	void TickForInference();
};
