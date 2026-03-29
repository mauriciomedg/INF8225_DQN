// Fill out your copyright notice in the Description page of Project Settings.


#include "MinionPLayer.h"

// Sets default values
AMinionPLayer::AMinionPLayer()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AMinionPLayer::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AMinionPLayer::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

// Called to bind functionality to input
void AMinionPLayer::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

}

