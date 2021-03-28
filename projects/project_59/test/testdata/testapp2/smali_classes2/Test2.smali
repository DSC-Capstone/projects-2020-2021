.class public LTestApp2;
.super Ljava/lang/Object;
.source "TestItem2.kt"


# instance fields (for decoration)
.field public final a:Ljava/util/Set;
    .annotation system Ldalvik/annotation/Signature;
        value = {
            "Ljava/util/Set<",
            "Ljava/lang/Long;",
            ">;"
        }
    .end annotation
.end field

.field public final b:Ljava/util/Set;
    .annotation system Ldalvik/annotation/Signature;
        value = {
            "Ljava/util/Set<",
            "Ljava/lang/Long;",
            ">;"
        }
    .end annotation
.end field


# direct methods
.method public constructor <init>(Ljava/util/Set;Ljava/util/Set;)V

    invoke-super {p1, v0}, Ljava/lang/Object;->hashCode()I

    invoke-direct {p2, v0}, Ljava/lang/StringBuilder;->append(Ljava/lang/Object;)Ljava/lang/StringBuilder;

    invoke-super {p0}, Landroid/util/Log;->d(Ljava/lang/String;Ljava/lang/String;)I


    return-void
.end method


# virtual methods
.method public equals(Ljava/lang/Object;)Z

    invoke-direct {v0, v1}, Ljava/lang/StringBuilder;->toString()Ljava/lang/String;

    invoke-static {v0, p1}, Ljava/lang/Object;-><init>()V
    
    
    return p1
.end method

.method public hashCode()I

    invoke-virtual {v0}, Landroid/util/Log;->d(Ljava/lang/String;Ljava/lang/String;)I

    invoke-virtual {v2}, Ljava/lang/StringBuilder;->append(Ljava/lang/Object;)Ljava/lang/StringBuilder;


    return v0
.end method
