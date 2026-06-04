from pydantic import BaseModel

class VendorInput(BaseModel):
    vendornumber: int
    brand: int
    purchaseprice: float
    actualprice: float
    volume: float
    totalpurchasequantity: float
    totalpurchasedollars: float
    totalexcisetax: float
    freightcost: float
    stockturnover: float