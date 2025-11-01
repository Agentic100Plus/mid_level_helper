from enum import Enum


class CareerLevel(Enum):
    JUNIOR = "주니어"
    MID = "중니어"
    SENIOR = "시니어"


class JobRole(Enum):
    BACKEND = "백엔드"
    FRONTEND = "프론트엔드"
    DEVOPS = "DevOps"
    IOS = "iOS"
    ANDROID = "AOS"
    FULLSTACK = "풀스택"
    DATA_ENGINEER = "데이터 엔지니어"
    ML_ENGINEER = "ML 엔지니어"
    ETC = "기타"
