import datetime

import pandas as pd


class DataManager:

    def __init__(self, year, league="Premier League"):
        """
        :param year: the season to consider, e.g. 2017 refers to the 2017/18 season.
            Can go back to 1993.

        :type year: int
        :param league: the league to consider.
            Can be one of "Premier League", "La Liga", "Serie A" or "Bundesliga".
        :type league: string.
        """
        now = datetime.datetime.now()
        if year < 1993:
            raise ValueError("Year must be 1993 or later.")
        elif year > now.year:
            raise ValueError("Year can't be in the future!")
        elif (year == now.year) and (now.month < 8):
            raise ValueError("The {} season hasn't started yet.".format(year))
        elif league not in ["Premier League", "La Liga", "Serie A", "Bundesliga"]:
            raise ValueError("{} unsupported.".format(league))

        self.year = year
        self.league = league
        self.raw_data = None
        self.clean_data = None
        self.teams = None

    def download_league_data(self):
        """Download the required data from www.football-data.co.uk."""
        year_str = str(self.year)[2:] + str(self.year + 1)[2:]
        league_map = {
            "Premier League": "E0",
            "La Liga": "SP1",
            "Serie A": "I1",
            "Bundesliga": "D1"
        }
        url = "http://www.football-data.co.uk/mmz4281/{}/{}.csv".format(
            year_str,
            league_map[self.league]
        )
        self.raw_data = pd.read_csv(url)

    def download_team_data(self):
        """Download team level data."""

    def clean_league_data(self):
        """Clean the raw downloaded data."""
        self.teams = self.raw_data['HomeTeam'].unique()
        self.clean_data = self.raw_data.copy()
        for i, team in enumerate(self.teams):
            home_idx = self.clean_data['HomeTeam'] == team
            away_idx = self.clean_data['AwayTeam'] == team
            self.clean_data.loc[home_idx, 'HomeTeam'] = i + 1
            self.clean_data.loc[away_idx, 'AwayTeam'] = i + 1
        self.clean_data[['HomeTeam', 'AwayTeam']] = self.clean_data[['HomeTeam', 'AwayTeam']].apply(pd.to_numeric)
        # for some reason, goals are not additionally counted as shots on target
        self.clean_data.loc[:, 'HST'] = self.clean_data['HST'] + self.clean_data['FTHG']
        self.clean_data.loc[:, 'AST'] = self.clean_data['AST'] + self.clean_data['FTAG']
        # similarly, shots on target are not classed as shots
        self.clean_data.loc[:, 'HS'] = self.clean_data['HST'] + self.clean_data['HS']
        self.clean_data.loc[:, 'AS'] = self.clean_data['AST'] + self.clean_data['AS']

