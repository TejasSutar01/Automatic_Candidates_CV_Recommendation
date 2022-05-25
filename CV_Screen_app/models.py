from django.db import models

# Create your models here.
class CV_Screen(models.Model):
    name = models.CharField(max_length=2000)
    Similarity_Percent = models.CharField(max_length=2000)
    Skills=models.CharField(max_length=2000)
    # def __str__(self):
    #     return str(self.name)
