from pydantic import BaseModel, Field, ConfigDict

class CaliforniaHousingInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    MedInc :int = Field(description="รายได้โดยประมาณของคนแถวนั้น (US Dollar)", examples=[20000], alias="People Income")
    AveRooms: int = Field(description="จำนวนห้อง", examples=[4], alias="Room size")
    AveBedrms: int = Field(description="จำนวนห้องนอน", examples=[2], alias="Bedroom size")
    Latitude: float = Field(description="พิกัด Latitude ที่ตั้งบ้าน", examples=[-122.23], alias="Latitude")
    Longtitude: float = Field(description="พิกัด Longtitude ที่ตั้งบ้าน", examples=[37.88], alias="Longtitude")