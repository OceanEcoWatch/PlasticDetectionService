import datetime

MANILLA_BAY_BBOX = (
    120.53058253709094,
    14.384463071206468,
    120.99038315968619,
    14.812423505754381,
)

LAST_WEEK_TIME_INTERVAL = (
    datetime.datetime.now() - datetime.timedelta(days=7),
    datetime.datetime.now(),
)
