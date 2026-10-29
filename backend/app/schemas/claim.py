from pydantic import BaseModel,Field
from typing import Literal

class ClaimCreate(BaseModel):
    claim_id:str
    amount:float=Field(ge=0)
    
class Claimstatusupdate(BaseModel):
    status:Literal["Pending", "Approved", "Rejected"]