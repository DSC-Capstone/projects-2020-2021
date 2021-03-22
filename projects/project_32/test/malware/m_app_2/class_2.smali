.class public Landroid/arch/core/internal/test_2;
.super Landroid/arch/core/internal/test;
.source "test_2.java"






.method public constructor <init>()V
    .locals 1

    .line 35
    invoke-direct {p0}, Landroid/SafeIterableMap;-><init>()V



    invoke-direct {v0}, Ljava/util/HashMap;-><init>()V
    return-void
.end method


# virtual methods
.method public sample(Ljava/lang/Object;)Ljava/util/test_method_1;


    invoke-virtual {v0, p1}, Ljava/util/HashMap;->get(Ljava/lang/Object;)Ljava/lang/Object;

.end method



.method public remove(Ljava/lang/Object;)Ljava/lang/Object;

    
   

    invoke-super {p0, p1}, Landroid/test_invoke;->remove(Ljava/lang/Object;)Ljava/lang/Object;


    invoke-virtual {v1, p1}, Ljava/util/HashMap;->remove(Ljava/lang/Object;)Ljava/lang/Object;

   
.end method


