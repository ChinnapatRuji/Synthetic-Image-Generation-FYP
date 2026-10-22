from fastapi import FastAPI

from app.api.router import api_router
from app.core.settings import settings


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    debug=settings.DEBUG,
)

app.include_router(api_router)


@app.get("/")
def root():
    return {
        "message": "Synthetic Image Generation API",
        "version": "0.1.0",
    }

original_openapi = app.openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = original_openapi()

    for definition in schema.get("components", {}).get("schemas", {}).values():
        for prop in definition.get("properties", {}).values():
            items = prop.get("items", {})

            if (
                items.get("type") == "string"
                and items.get("contentMediaType")
                == "application/octet-stream"
            ):
                items.pop("contentMediaType", None)
                items["format"] = "binary"

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi