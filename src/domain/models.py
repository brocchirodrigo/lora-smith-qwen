from pydantic import BaseModel


class WPRendered(BaseModel):
    """Campo com valor renderizado da API WordPress."""

    rendered: str
    protected: bool = False


class WPPost(BaseModel):
    """Representa um post da API REST do WordPress."""

    id: int
    status: str
    link: str
    title: WPRendered
    content: WPRendered
