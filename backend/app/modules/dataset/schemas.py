from uuid import UUID
from pydantic import BaseModel, Field, field_validator

class ClassGeneration(BaseModel):
    prompt: str = Field(default='', max_length=2000)
    negative_prompt: str = Field(default='', max_length=2000)

class ClassCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default='', max_length=2000)
    generation: ClassGeneration = Field(default_factory=ClassGeneration)

    @field_validator('name')
    @classmethod
    def validate_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError('Name cannot be empty')
        return value

class ClassUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=2000)
    generation: ClassGeneration | None = None

    @field_validator('name')
    @classmethod
    def validate_name(cls, value):
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError('Name cannot be empty')
        return value

class ClassResponse(BaseModel):
    id: UUID
    name: str
    image_count: int = 0
    description: str = ''
    generation: ClassGeneration = Field(default_factory=ClassGeneration)

class DatasetResponse(BaseModel):
    classes: list[ClassResponse] = Field(default_factory=list)
