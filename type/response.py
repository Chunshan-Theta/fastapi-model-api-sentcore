from typing import TypedDict, NewType

modelResult = NewType("result", list)


class JsonResponBase(TypedDict):
    status: int


class JsonResponMsg(JsonResponBase):
    result: modelResult
