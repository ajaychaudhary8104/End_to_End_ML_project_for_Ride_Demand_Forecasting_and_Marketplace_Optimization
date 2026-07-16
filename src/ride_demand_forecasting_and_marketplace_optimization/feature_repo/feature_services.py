from feast import FeatureService

from feature_views import ride_feature_view


forecast_service = FeatureService(

    name="forecast_service",

    features=[
        ride_feature_view
    ]
)