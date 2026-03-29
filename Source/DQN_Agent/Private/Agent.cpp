// Fill out your copyright notice in the Description page of Project Settings.


#include "Agent.h"

#include "DQNComponent.h"
#include "Components/StaticMeshComponent.h"
#include "UObject/ConstructorHelpers.h"

// Sets default values
AAgent::AAgent()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	SphereComp = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("SphereComp"));
	SetRootComponent(SphereComp);

	// Assign the Engine's built-in sphere mesh
	static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereMesh(TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	if (SphereMesh.Succeeded())
	{
		Cast<UStaticMeshComponent>(SphereComp)->SetStaticMesh(SphereMesh.Object);
	}

	// Reasonable defaults
	SphereComp->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
	SphereComp->SetCollisionProfileName(TEXT("PhysicsActor")); // good default for sim bodies
	SphereComp->SetSimulatePhysics(true);
	SphereComp->SetEnableGravity(true);

	// Optional tuning
	SphereComp->SetNotifyRigidBodyCollision(true); // hit events
	SphereComp->SetLinearDamping(0.05f);
	SphereComp->SetAngularDamping(0.1f);

	DQNComp = CreateDefaultSubobject<UDQNComponent>(TEXT("DQNComponent"));

}

// Called when the game starts or when spawned
void AAgent::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AAgent::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

