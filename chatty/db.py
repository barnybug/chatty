from sqlalchemy import create_engine
from . import models


def engine():
    ret = create_engine("sqlite:///chat.db")
    models.Base.metadata.create_all(ret)
    return ret
