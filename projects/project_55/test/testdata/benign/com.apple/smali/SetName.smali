.class Lcom/apple/SetName;
.super Lcom/apple/setter/target/SetContent;
.source "SetName.java"

.method public onSet(int;)V

	.locals 1

	.param p1 "inputNum"

	invoke-super{p0, p1}, Lcom/apple/setter/target/SetContent;->OnSet(int;)V

	const-string v0, "Failed to set"
	invoke-direct {p0, v0}, Lcom/exception/MissingComponentException;-><init>(Ljava/lang/String;)V

	return-void
.end method
