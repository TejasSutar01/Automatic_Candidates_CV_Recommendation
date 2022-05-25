from rest_framework import serializers
from CV_Screen_app.models import CV_Screen

class CV_ScreenSerializer(serializers.ModelSerializer):
    class Meta:
        model=CV_Screen
        fields=('__all__')