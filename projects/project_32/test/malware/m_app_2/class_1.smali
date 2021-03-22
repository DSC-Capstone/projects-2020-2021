.class public Landroid/arch/core/internal/test_1;
.super Landroid/arch/core/internal/test;
.source "test_1.java"



.method public constructor <init>()V
    .locals 1

    .line 35
    invoke-direct {p0}, Landroid/test;-><init>()V

    .line 37
    new-instance v0, Ljava/util/HashMap;

    invoke-direct {v0}, Ljava/util/HashMap;-><init>()V


    return-void
.end method


# virtual methods
.method public test_a(Ljava/lang/Object;)Ljava/util/test_method_1;
    .locals 1
    .annotation system Ldalvik/annotation/Signature;
        value = {
            "(TK;)",
            "Ljava/util/test_call<",
            "TK;TV;>;"
        }
    .end annotation

    invoke-virtual {v0, p1}, Ljava/util/HashMap;->get(Ljava/lang/Object;)Ljava/lang/Object;

.end method



.method public test_m(Ljava/lang/Object;)Ljava/lang/Object;
    .locals 2
    
    .annotation system Ldalvik/annotation/Signature;
        value = {
            "(TK;)TV;"
        }
    .end annotation

    .line 56
    invoke-super {p0, p1}, Landroid/test_invoke;->remove(Ljava/lang/Object;)Ljava/lang/Object;

    .line 57
    invoke-virtual {v1, p1}, Ljava/util/HashMap;->remove(Ljava/lang/Object;)Ljava/lang/Object;
    invoke-virtual {v1, p1}, Ljava/util/HashMap;->remove(Ljava/lang/Object;)Ljava/lang/Object;


    return-object v0
.end method


