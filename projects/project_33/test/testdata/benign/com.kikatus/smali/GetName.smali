.class Lcom/kaktus/GetName;
.super Lcom/kaktus/getter/target/ViewContent;
.source "GetName.java"

.method public constructor <init>(Landrioid/view/View;)V
	.locals 2

	.param p1, "view"
	const-string v0, "Failed to get target"

	invoke-direct {p0, p1}, Lcom/kaktus/getter/target/ViewContent;-><init>(Landrioid/view/View;)V
	
	invoke-direct {p0, v0}, Lcom/exception/MissingComponentException;-><init>(Ljava/lang/String;)V

	invoke-static {p0}, Ljava/lang.Objetc;->get()Ljava/lang/String;
	return-void
.end method
