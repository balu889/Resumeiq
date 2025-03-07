from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    is_admin = models.BooleanField(default=False)
    is_user = models.BooleanField(default=False)
    is_service_provider = models.BooleanField(default=False)

    class Meta:
        swappable = 'AUTH_USER_MODEL'
