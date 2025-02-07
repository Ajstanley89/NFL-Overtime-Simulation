import numpy as np
import pandas as pd

from scipy import stats

class FootballTeam():
    """Object describing a football team for this simulation. 
    Downs will index from 0, so down=3 is fourth down. Think of it as "number of downs completed"
    """
    def __init__(self, name: str, 
                 has_completed_a_posession:bool = False,
                 min_yards:int | float=-15, 
                 exp_yards:int | float=6, 
                 max_yards:int | float=25, 
                 punt_distance:int | float=45, 
                 punt_stdv:int | float=4,
                 fourth_down_yard_thresh: int | float= 2, 
                 fourth_down_position_thresh: int | float= 45,
                 two_point_conversion_rate: float = 0.5,
                 go_for_two: bool = False,
                 need_touchdown:bool=False,
                 need_score:bool=False):

        self.name = name
        self.has_completed_a_posession = has_completed_a_posession
        self.min_yards = min_yards
        self.exp_yards = exp_yards
        self.max_yards = max_yards
        self.punt_distance = punt_distance
        self.punt_stdv = punt_stdv
        self.fourth_down_yard_thresh = fourth_down_yard_thresh
        self.fourth_down_position_thresh = fourth_down_position_thresh
        self.two_point_conversion_rate = two_point_conversion_rate
        self.go_for_two = go_for_two
        self.need_touchdown = need_touchdown
        self.need_score = need_score

        self.downs_completed = 0
        self.field_position:int | float = 25
        self.score: int = 0
        self.first_down_yards_needed: int | float = 10,
        self.had_ball_first: bool = None
        self.has_completed_a_possession = False
        self.drive_data = []

    def update_score(self, score_value:int):
        self.score += score_value

    def set_field_position(self, new_field_position:int):
        self.field_position = new_field_position

    def update_field_position(self, play_result:int | float):
        self.field_position += play_result

    def update_yards_to_firstdown(self, play_result: int | float):
        self.first_down_yards_needed -= play_result

        if self.first_down_yards_needed <= 0:
                self.reset_series()

    def set_down(self, new_down:int):
        self.downs_completed = new_down

    def reset_series(self):
        self.downs_completed = 0
        self.first_down_yards_needed = 10

    def turnover_possession(self):
        self.set_offense(False)
        self.reset_series()

    def set_need_touchdown(self, value:bool):
        self.need_touchdown = value

    def set_need_score(self, value:bool):
        self.need_score = value

    def set_has_completed_possession(self, value:bool):
        self.has_completed_a_possession = value

    def set_had_ball_first(self, value: bool):
        self.had_ball_first = value

    def update_drive_data(self, df: pd.DataFrame):
        self.drive_data.append(df)

    def set_two_point_conversion_rate(self, value: float):
        self.two_point_conversion_rate = value

    def set_go_for_two(self, value: bool):
        self.go_for_two = value

    def get_drive_data(self):
        return self.drive_data

    def check_touchdown(self):
        return self.field_position >= 100

    def get_play_conditions_dict(self, play_type:str = None) -> dict:
        """Returns a boiler plate dict with game-state information on the start of any play"""
        
        result_dict = {'field_position': self.field_position,
                       'down': self.downs_completed,
                       'yards_to_first_down': self.first_down_yards_needed}

        if play_type is not None:
            result_dict['play_type'] = play_type

        return result_dict

    def simulate_play_yards(self, n_plays: int=1) -> float | np.ndarray:
        play_result = np.random.triangular(self.min_yards, self.exp_yards, self.max_yards, n_plays)

        if n_plays == 1:
            play_result = play_result[0]

        return play_result
    
    def touch_down_conversion(self):
        points_from_play = 0

        if not self.go_for_two:
            points_from_play += 1

        elif np.random.uniform() <= self.two_point_conversion_rate:
            points_from_play += 2

        return points_from_play
    
    def get_last_play(self):
        return self.drive_data[-1].iloc[-1, :]
        
    def run_play(self, debug=False) -> dict:
        """Simulates play with yardarge results by sampling from triangular distribution. Relies on user-supplied minimum likely yardage, most-liekly yardage, and max likely yardage."""
        play_result = self.simulate_play_yards()

        result_dict = self.get_play_conditions_dict('run_play')

        self.update_field_position(play_result)
        self.downs_completed += 1
        points_from_play = 0
        
        if self.check_touchdown():
            points_from_play = 6
            points_from_play += self.touch_down_conversion()
            self.update_score(points_from_play)

        else:
            self.update_yards_to_firstdown(play_result)

        result_dict['yards_gained'] = play_result
        result_dict['points_from_play'] = points_from_play
        
        return result_dict
    
    def punt(self) -> dict:
        """Simulates a punt with normal distibution. Google search shows average punt distance is ~45 yards with a stdev of ~4 yards. 
        
        Does not account for runbacks.
        """
        punt_distance = np.random.normal(self.punt_distance, self.punt_stdv)
        punt_landing_spot = punt_distance + self.field_position
        
        # account for touchback
        opponent_position = 100 - punt_landing_spot if punt_landing_spot <= 100 else 25

        result_dict = self.get_play_conditions_dict('punt')
        result_dict['punt_distance'] = punt_distance
        result_dict['opponent_position'] = opponent_position
        result_dict['points_from_play'] = 0

        self.downs_completed += 1

        return result_dict
            
    def kick_field_goal(self) -> dict:
        """Simulates field goal success"""
        # simulate kick
        kick_attempt = np.random.uniform()

        points_from_play = 0

        if kick_attempt <= self.get_field_goal_probability():
            points_from_play = 3
            self.update_score(points_from_play)

        result_dict = self.get_play_conditions_dict('field_goal')
        result_dict['yards_gained'] = -5 # accounts for hold distance in case field goal is missed
        result_dict['made_kick'] = kick_attempt <= self.get_field_goal_probability()
        result_dict['points_from_play'] = points_from_play

        self.downs_completed += 1

        return result_dict

    def get_field_goal_probability(self) -> float:
        """Define short, medium, and long ranges for field goal
        Hard coded for now, but we can adapt this later to take user inputs.
        """

        yards_to_endzone = 100 - self.field_position
        additional_kick_yards = 17 # accounts for endzone length and holdign distance from line of scrimmage
        kick_distance = yards_to_endzone + additional_kick_yards

        # baseline kick probability will decrease based on distance
        # TODO: modify to accept user-supplied function determining kick probability
        if kick_distance <= 40:
            kick_probability = 0.90
            
        elif kick_distance <= 50:
            kick_probability = 0.70
            
        elif kick_distance  <= 60:
            kick_probability = 0.60
            
        else:
            kick_probability = 0

        return kick_probability

    def meets_fouth_down_criteria(self):
        return self.field_position > self.fourth_down_position_thresh and self.first_down_yards_needed < self.fourth_down_yard_thresh
        
    def fourth_down_decision(self):
        if self.need_touchdown or self.meets_fouth_down_criteria():
            return self.run_play()
        
        elif self.get_field_goal_probability() > .55:
            return self.kick_field_goal()
        
        # need to go for it on 4th down if you need a score but aren't in FG range
        elif self.need_score:
            return self.run_play()

        else:
            return self.punt()

    def simulate_drive(self, starting_position: int | float=25, debug=False) -> pd.DataFrame:
        self.set_field_position(starting_position)
        self.reset_series()

        plays = []

        while self.downs_completed <= 3 and self.field_position < 100:
            # always run a play if it's not 4th down
            if self.downs_completed <= 2:
                result = self.run_play()

            else:
                result = self.fourth_down_decision()

            if debug:
                print(f'Play Result: {result}')
                
            plays.append(result)

        self.set_has_completed_possession(True)
        df = pd.DataFrame(plays)
        self.update_drive_data(df)

        return self

    def simulate_series(self, n_plays=3) -> float:
        """Simulates 3 plays. Can be used to simulate first down probability for a series"""
        series_yardage = self.simulate_play_yards(n_plays=n_plays)

        return series_yardage.sum()
    
    def determine_opponent_position(self) -> int | float:
        """Gets the last play from the team's drive and determines where the opponent should start"""
        last_play = self.get_last_play()
        play_type = last_play.play_type

        if last_play.points_from_play > 0:
            return 25 # no variation on kickoffs for now
            
        elif play_type == 'punt':
            return last_play.opponent_position

        else:
            # turnover on downs. Flip the field from the last play
            return 100 - (last_play.field_position + last_play.yards_gained)
        
class FootballGame():
    def __init__(self, ball_first_team:FootballTeam, ball_second_team:FootballTeam):
        self.ball_first_team = ball_first_team
        self.ball_second_team = ball_second_team
        self.game_data: dict = {}

        self.winner = None
        self.loser = None

    def check_for_winner(self):
        return all([self.ball_first_team.has_completed_a_possession, self.ball_second_team.has_completed_a_possession]) and self.ball_first_team.score != self.ball_second_team.score

    def set_winning_team(self):
        # redundant check if there is truly a winner
        if not self.check_for_winner:
            print('No winner yet!')
            
        elif self.ball_first_team.score > self.ball_second_team.score:
            self.winner = self.ball_first_team
            self.loser = self.ball_second_team
            
        elif self.ball_first_team.score < self.ball_second_team.score:
            self.winner = self.ball_second_team
            self.loser = self.ball_first_team

        else:
            print('Something wierd happened!')
            self.winner = None
        
    def get_last_play(self, ball_first_team:True):
        if ball_first_team:
            return self.ball_first_team.get_drive_data()[-1].iloc[-1, :]
        else:
            return self.ball_second_team.get_drive_data()[-1].iloc[-1, :]

    def update_game_data(self):
        """Updates game data dict fith high-level game information"""

        if self.winner is not None:
            winner_df = pd.concat(self.winner.get_drive_data())
            loser_df = pd.concat(self.loser.get_drive_data())

            self.game_data['winner'] = self.winner.name
            self.game_data['loser'] = self.loser.name
            self.game_data['winner_score'] = self.winner.score
            self.game_data['winner_score'] = self.loser.score
            self.game_data['winner_had_ball_first'] = self.winner.had_ball_first
            self.game_data['winning_team_number_of_drives'] = len(self.winner.get_drive_data())
            self.game_data['losing_team_number_of_drives'] = len(self.loser.get_drive_data())
            self.game_data['total_yards_winning_team'] = winner_df.sum().yards_gained
            self.game_data['total_yards_losing_team'] = loser_df.sum().yards_gained


        
    def determine_drive_start(self, ball_first_team=True):
        last_play = self.get_last_play(ball_first_team)
        play_type = last_play.play_type

        if last_play.points_from_play > 0:
            return 25 # no variation on kickoffs for now
            
        elif play_type == 'punt':
            return last_play.opponent_position

        else:
            # turnover on downs. Flip the field from the last play
            return 100 - (last_play.field_position + last_play.yards_gained)
        
    def enable_sudden_death(self):
        self.ball_first_team.set_need_touchdown(False)
        self.ball_first_team.set_need_score(False)
        self.ball_second_team.set_need_touchdown(False)
        self.ball_second_team.set_need_score(False)
        

    def simulate_game(self):
        # set team ball states
        self.ball_first_team.set_had_ball_first(True)
        self.ball_second_team.set_had_ball_first(False)

        # check for starting position
        starting_position = 25

        # first team simulate a drive
        self.ball_first_team.simulate_drive(starting_position).determine_opponent_position()
        starting_position = self.ball_first_team.determine_opponent_position()

        # update whether second team needs touchdown or field goal 
        if self.ball_first_team.score == 7:
            self.ball_second_team.set_need_touchdown(True)
        
        if self.ball_first_team.score == 3:
            self.ball_second_team.set_need_score(True)

        # next team simulates drive
        self.ball_second_team.simulate_drive(starting_position)
        starting_position = self.ball_second_team.determine_opponent_position()

        # play sudden death until there's a score. Might loop might not execute at all
        self.enable_sudden_death()

        while not self.check_for_winner():
            self.ball_first_team.simulate_drive(starting_position)

            if self.check_for_winner():
                break
    
            starting_position = self.ball_first_team.determine_opponent_position()
            self.ball_second_team.simulate_drive(starting_position)
            starting_position = self.ball_second_team.determine_opponent_position()

        self.set_winning_team()
        self.update_game_data()


        return self
    
team1_params = {'name': 'receiving_team',
                'go_for_two': False}

team2_params = {'name': 'kicking_team',
                'go_for_two': False}

class FootballSimulation():
    '''Wrapper to hold many simulations of games vs two teams'''
    def __init__(self, team1_params: dict, team2_params: dict):
        self.team1_params = team1_params
        self.team2_params = team2_params

        self.games: list = None

    def simulate_new_game(self):
        team1 = FootballTeam(**self.team1_params)
        team2 = FootballTeam(**self.team2_params)
    
        game = FootballGame(team1, team2)
    
        return game.simulate_game()

    def simulate_games(self, n_interations: int = 1000):

        self.games = [self.simulate_new_game() for i in range(n_interations)]

        return self

    def summarize_df(self) -> pd.DataFrame:
        return pd.DataFrame([game.game_data for game in self.games])

             
if __name__ == '__main__':
    from pprint import pprint

    np.random.seed(123)
    # test team class
    team = FootballTeam('team1', exp_yards=6, min_yards=-15, max_yards=25)

    n_drives = 10000
    for i in range(n_drives):
        team.simulate_drive(25, False)

    last_plays_df = pd.concat([df.iloc[-1:, :] for df in team.get_drive_data()])
    score_type_ratio = last_plays_df.points_from_play.value_counts() / n_drives

    # test actual simulated game
    print('Scores Breaking Down From Simulated Drives:')
    print(score_type_ratio)

    # simualte several games
    for i in range(5):
        
        team1 = FootballTeam('The Idaho Beets')
        team2 = FootballTeam('The Boston Wicked Tuna', go_for_two=True)

        print(f'Simulating Football Game: {team1.name} vs {team2.name}')

        game = FootballGame(team1, team2)
        game.simulate_game()

        pprint(game.game_data)
        print('-' * 50)

    team1_params = {'name': 'receiving_team',
                'go_for_two': False}

    team2_params = {'name': 'kicking_team',
                    'go_for_two': False}

    # test smualtion wrapper class
    print('Testing FootballSimulation Wrapper...')
    test = FootballSimulation(team1_params, team2_params)
    test.simulate_games(5)

    pprint(test.summarize_df())