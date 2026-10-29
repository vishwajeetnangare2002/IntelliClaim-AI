
from sqlalchemy import Column, Integer, String, Float
from app.database.db import Base


class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(String, unique=True, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String,default="Pending",nullable=False)
        
        
        
    