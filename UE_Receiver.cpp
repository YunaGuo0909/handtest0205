// ============================================================
// UE5 接收手势数据的示例代码
// 复制到你的 Actor 类中使用
// ============================================================

// ---------- 头文件需要添加 ----------
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "Json.h"

// ---------- 头文件中声明 ----------
UPROPERTY()
FSocket* Socket;

UPROPERTY(BlueprintReadOnly)
FString CurrentGesture;  // 当前手势

UPROPERTY(BlueprintReadOnly)
int32 NumHands;  // 手的数量

// ---------- BeginPlay 中初始化 ----------
void AYourActor::BeginPlay()
{
    Super::BeginPlay();
    
    // 创建 UDP Socket
    ISocketSubsystem* SocketSub = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    Socket = SocketSub->CreateSocket(NAME_DGram, TEXT("HandSocket"), false);
    Socket->SetNonBlocking(true);
    Socket->SetReuseAddr(true);
    
    // 绑定端口 5000
    TSharedRef<FInternetAddr> Addr = SocketSub->CreateInternetAddr();
    Addr->SetAnyAddress();
    Addr->SetPort(5000);
    Socket->Bind(*Addr);
    
    UE_LOG(LogTemp, Log, TEXT("Hand tracking receiver started on port 5000"));
}

// ---------- Tick 中接收数据 ----------
void AYourActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    
    if (!Socket) return;
    
    uint32 Size;
    while (Socket->HasPendingData(Size))
    {
        TArray<uint8> Data;
        Data.SetNumUninitialized(FMath::Min(Size, 65535u));
        int32 Read;
        
        if (Socket->Recv(Data.GetData(), Data.Num(), Read))
        {
            Data.Add(0);
            FString Json = UTF8_TO_TCHAR(Data.GetData());
            ParseHandData(Json);
        }
    }
}

// ---------- 解析 JSON ----------
void AYourActor::ParseHandData(const FString& Json)
{
    TSharedPtr<FJsonObject> Obj;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Json);
    
    if (!FJsonSerializer::Deserialize(Reader, Obj)) return;
    
    NumHands = Obj->GetIntegerField("num_hands");
    
    const TArray<TSharedPtr<FJsonValue>>* Hands;
    if (Obj->TryGetArrayField("hands", Hands) && Hands->Num() > 0)
    {
        auto Hand = (*Hands)[0]->AsObject();
        CurrentGesture = Hand->GetStringField("gesture");
        
        // 根据手势做不同的事
        if (CurrentGesture == "fist")
        {
            // 握拳
        }
        else if (CurrentGesture == "open")
        {
            // 张开
        }
        else if (CurrentGesture == "peace")
        {
            // 比耶
        }
        else if (CurrentGesture == "thumb_up")
        {
            // 点赞
        }
        else if (CurrentGesture == "pointing")
        {
            // 指向
        }
        // ... 其他手势
    }
}

// ---------- EndPlay 中清理 ----------
void AYourActor::EndPlay(const EEndPlayReason::Type Reason)
{
    if (Socket)
    {
        Socket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Socket);
    }
    Super::EndPlay(Reason);
}

// ---------- Build.cs 需要添加的模块 ----------
// PublicDependencyModuleNames.AddRange(new string[] { 
//     "Sockets", "Networking", "Json", "JsonUtilities" 
// });
