from fastapi import FastAPI,Depends,HTTPException,Query,UploadFile,File
from pydantic import BaseModel,Field
from sqlalchemy.orm import Session

from app.database.db import SessionLocal,engine, Base
from app.models.claim import Claim as ClaimModel
from app.services.ocr_service import extract_text_from_pdf
from app.schemas.claim import ClaimCreate,Claimstatusupdate
from app.services.claim_extractor import extract_claim_details

from typing import Literal


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
    
    


@app.post("/documents/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file"
        )

    pdf_bytes = await file.read()

    if len(pdf_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="PDF file must be 10 MB or smaller"
        )

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty"
        )

    try:
        extracted_text = extract_text_from_pdf(pdf_bytes)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF. Please upload a valid PDF file."
        )

    if not extracted_text:
        raise HTTPException(
            status_code=400,
            detail="No embedded text found. This PDF may require OCR."
        )

    claim_details = extract_claim_details(extracted_text)

    if not claim_details["is_valid"]:
        return {
            "message": "Claim validation failed",
            "claim_details": claim_details
        }

    existing_claim = db.query(ClaimModel).filter(
        ClaimModel.claim_id == claim_details["claim_id"]
    ).first()

    if existing_claim:
        raise HTTPException(
            status_code=409,
            detail=f"Claim ID {claim_details['claim_id']} already exists"
        )

    new_claim = ClaimModel(
        claim_id=claim_details["claim_id"],
        amount=claim_details["claim_amount"],
        status="Pending"
    )

    db.add(new_claim)

    try:
        db.commit()
        db.refresh(new_claim)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not save the claim to the database"
        )

    return {
        "message": "Claim extracted and saved successfully",
        "filename": file.filename,
        "claim_details": claim_details,
        "saved_claim": {
            "id": new_claim.id,
            "claim_id": new_claim.claim_id,
            "amount": new_claim.amount,
            "status": new_claim.status
        }
    }
