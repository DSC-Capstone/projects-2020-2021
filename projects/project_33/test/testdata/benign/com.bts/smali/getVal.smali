.class public interface;
.super Lcom/apple/setter/target/SetContent;
.source "snapchat.java"

.method public getVal(str);)V

	.param p3 "index"
	invoke-static {p0}, Ljava/lang/Object;->get()Ljava/lang/String;

	invoke-static{p3}, Landroid/os/Bundle;->getVal(str;)V

	const-string v0, "value"
	invoke-direct {p3, v0}, Lcom/exception/MissingComponentException;-><init>(Ljava/lang/String;)V

	return-void
.end method

.method public getVal(str);)V

	.param p3 "index"
	invoke-static {p0}, Ljava/lang/Object;->get()Ljava/lang/String;

	invoke-static{p3}, Landroid/os/Bundle;->getVal(str;)V

	const-string v0, "value"
	invoke-direct {p3, v0}, Lcom/exception/MissingComponentException;-><init>(Ljava/lang/String;)V

	return-void
.end method
