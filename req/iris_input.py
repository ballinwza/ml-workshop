from pydantic import BaseModel, Field

class IrisInput(BaseModel):
    sepal_length: float = Field(description="ความยาวกลีบเลี้ยง (ซม.)", examples=[5.1])
    sepal_width: float = Field(description="ความกว้างกลีบเลี้ยง (ซม.)", examples=[3.5])
    petal_length: float = Field(description="ความยาวกลีบดอก (ซม.)", examples=[1.4])
    petal_width: float = Field(description="ความกว้างกลีบดอก (ซม.)", examples=[0.2])
