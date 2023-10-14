import datetime


def get_today_str():
    return datetime.datetime.today().strftime("%Y-%m-%d")


def get_past_date(days):
    return (datetime.datetime.today() - datetime.timedelta(days=days)).strftime(
        "%Y-%m-%d"
    )
