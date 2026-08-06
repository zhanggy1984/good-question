"""仪表盘 API"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from middleware.auth import get_current_user
from models import User
from schemas.dashboard import DashboardResponse
from services import dashboard_service

router = APIRouter()


@router.get("", response_model=DashboardResponse)
def dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """仪表盘统计（所有登录用户）"""
    return dashboard_service.get_stats(db)
