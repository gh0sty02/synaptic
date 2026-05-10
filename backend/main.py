from fastapi import FastAPI

app = FastAPI(title="Synaptic")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
