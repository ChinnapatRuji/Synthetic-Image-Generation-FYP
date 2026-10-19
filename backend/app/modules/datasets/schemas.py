from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class ClassCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Name cannot be empty")
        return value


class ClassUpdate(ClassCreate):
    pass


class ClassResponse(BaseModel):
    id: UUID
    name: str
    image_count: int = 0


class DatasetResponse(BaseModel):
    classes: list[ClassResponse] = Field(default_factory=list)
