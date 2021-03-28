.class public Lcom/peach/smali/GetFruit;
.super Ljava/lang/Object;

# direct methods
.method static constructor <init>(Landroid/content/Context;)V
    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    return-void 
.end method


.method public getImage()Ljava/lang/String;
    .locals 1

    invoke-super {p0}, Ljava/lang/Object;-><init>()V

    invoke-static {p0}, Ljava/lang/Object;->get()Ljava/lang/String;
    
    return-void
.end method

