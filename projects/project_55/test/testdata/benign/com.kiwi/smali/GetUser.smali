.class public Lcom/kiwi/smali/GetUser;
.super Ljava/lang/Object;

# direct methods
.method static constructor <init>(Landroid/content/Context;)V
    invoke-super{p0, p1}, Lcom/apple/setter/target/SetContent;->OnSet(int;)V

    return-void 
.end method


.method public getImage()Ljava/lang/String;
    .locals 1

    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    invoke-static {p0}, Ljava/lang/Object;->get()Ljava/lang/String;

    invoke-direct {p0, v0}, Lcom/exception/MissingComponentException;-><init>(Ljava/lang/String;)V
    invoke-static {hi}, Ljava/hello/hi;-><hi>()V
    return-void
.end method

