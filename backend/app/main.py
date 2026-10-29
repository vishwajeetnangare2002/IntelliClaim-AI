from fastapi import FastAPI,Depends,HTTPException,Query
from pydantic import BaseModel,Field
from sqlalchemy.orm import Session

from app.database.db import SessionLocal,engine, Base
from app.models.claim import Claim as ClaimModel

from typing import Literal
from app.schemas.claim import ClaimCreate,Claimstatusupdate

app=FastAPI()
Base.metadata.create_all(bind=engine)

def get_db():
    db=SessionLocal()
    
    try:
        yield db
    finally:
        db.close()
        

@app.get("/")
def home():
    return{
        "message":"home page created successfully"
    }
    
@app.get("/health")
def get_health():
    return{
        "status":"healthy",
        "project":"insurance-claim-ai"
    }
 
 
class Claim(BaseModel):
    claim_id: str
    amount: float=Field(ge=0)  
     
@app.post("/claims")
def create_claim(claim: ClaimCreate,db:Session=Depends(get_db)):
    db_claim = ClaimModel(
        claim_id=claim.claim_id,
        amount=claim.amount
    )
    
    db.add(db_claim)
    db.commit()
    db.refresh(db_claim)
   
    return {
        "id": db_claim.id,
        "claim_id": db_claim.claim_id,
        "amount": db_claim.amount,
        "status": db_claim.status,
        "message": "Claim saved successfully"
    }
    
@app.get("/claims")
def get_claims(db: Session = Depends(get_db)):
    claims = db.query(ClaimModel).all()

    return claims
   
@app.get("/claims/{claim_id}")
def get_claim(claim_id: str, db: Session = Depends(get_db)):
    claim = db.query(ClaimModel).filter(
        ClaimModel.claim_id == claim_id
    ).first()

    if claim is None:
        raise HTTPException(
        status_code=404,
        detail="Claim not found"
    )

    return claim

@app.put("/claims/{claim_id}")
def update_claim(
    claim_id: str,
    amount: float = Query(ge=0),
    db: Session = Depends(get_db)
):
    claim = db.query(ClaimModel).filter(
        ClaimModel.claim_id == claim_id
    ).first()

    if claim is None:
        raise HTTPException(
            status_code=404,
            detail="Claim not found"
        )

    claim.amount = amount
    db.commit()
    db.refresh(claim)

    return claim

@app.delete("/claims/{claim_id}")
def delete_claim(
    claim_id: str,
    db: Session = Depends(get_db)
):
    claim = db.query(ClaimModel).filter(
        ClaimModel.claim_id == claim_id
    ).first()

    if claim is None:
        raise HTTPException(
            status_code=404,
            detail="Claim not found"
        )

    db.delete(claim)
    db.commit()

    return {"message": "Claim deleted successfully"}

@app.patch("/claims/{claim_id}/status")
def update_claim_status(
    claim_id: str,
    data:Claimstatusupdate,
    db: Session = Depends(get_db)
):
    claim = db.query(ClaimModel).filter(
        ClaimModel.claim_id == claim_id
    ).first()

    if claim is None:
        raise HTTPException(
            status_code=404,
            detail="Claim not found"
        )

    claim.status = data.status
    db.commit()
    db.refresh(claim)

    return {
        "claim_id": claim.claim_id,
        "status": claim.status,
        "message": "Claim status updated successfully"
    }