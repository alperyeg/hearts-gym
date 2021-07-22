"""
The reward function an agent optimizes to win at Hearts.
"""

import numpy as np

from hearts_gym.utils.typing import Reward
from .hearts_env import HeartsEnv


def get_card_name(card):
    if not card:
        return None

    if card.suit == 3:
        # Spades
        if card.rank == 12:
            return 'ace_spades'
        elif card.rank == 11:
            return 'king_spades'
        elif card.rank == 10:
            return 'queen_spades'
        elif card.rank == 9:
            return 'jack_spades'
        elif card.rank == 8:
            return 'ten_spades'
        elif card.rank == 7:
            return 'nine_spades'
        elif card.rank == 6:
            return 'eight_spades'
        elif card.rank == 5:
            return 'seven_spades'
        elif card.rank == 4:
            return 'six_spades'
        elif card.rank == 3:
            return 'five_spades'
        elif card.rank == 2:
            return 'four_spades'
        elif card.rank == 1:
            return 'three_spades'
        elif card.rank == 0:
            return 'two_spades'
    elif card.suit == 0:
        # Clubs
        if card.rank == 12:
            return 'ace_clubs'
        elif card.rank == 11:
            return 'king_clubs'
        elif card.rank == 10:
            return 'queen_clubs'
        elif card.rank == 9:
            return 'jack_clubs'
        elif card.rank == 8:
            return 'ten_clubs'
        elif card.rank == 7:
            return 'nine_clubs'
        elif card.rank == 6:
            return 'eight_clubs'
        elif card.rank == 5:
            return 'seven_clubs'
        elif card.rank == 4:
            return 'six_clubs'
        elif card.rank == 3:
            return 'five_clubs'
        elif card.rank == 2:
            return 'four_clubs'
        elif card.rank == 1:
            return 'three_clubs'
        elif card.rank == 0:
            return 'two_clubs'
    elif card.suit == 2:
        # Hearts
        if card.rank == 12:
            return 'ace_hearts'
        elif card.rank == 11:
            return 'king_hearts'
        elif card.rank == 10:
            return 'queen_hearts'
        elif card.rank == 9:
            return 'jack_hearts'
        elif card.rank == 8:
            return 'ten_hearts'
        elif card.rank == 7:
            return 'nine_hearts'
        elif card.rank == 6:
            return 'eight_hearts'
        elif card.rank == 5:
            return 'seven_hearts'
        elif card.rank == 4:
            return 'six_hearts'
        elif card.rank == 3:
            return 'five_hearts'
        elif card.rank == 2:
            return 'four_hearts'
        elif card.rank == 1:
            return 'three_hearts'
        elif card.rank == 0:
            return 'two_hearts'
    elif card.suit == 1:
        # Diamond
        if card.rank == 12:
            return 'ace_diamond'
        elif card.rank == 11:
            return 'king_diamond'
        elif card.rank == 10:
            return 'queen_diamond'
        elif card.rank == 9:
            return 'jack_diamond'
        elif card.rank == 8:
            return 'ten_diamond'
        elif card.rank == 7:
            return 'nine_diamond'
        elif card.rank == 6:
            return 'eight_diamond'
        elif card.rank == 5:
            return 'seven_diamond'
        elif card.rank == 4:
            return 'six_diamond'
        elif card.rank == 3:
            return 'five_diamond'
        elif card.rank == 2:
            return 'four_diamond'
        elif card.rank == 1:
            return 'three_diamond'
        elif card.rank == 0:
            return 'two_diamond'
    return None


def check_relevant_cards(cards_list):
    cards = {
        'ace_clubs': False,  # Clubs
        'king_clubs': False,
        'queen_clubs': False,
        'jack_clubs': False,
        'ten_clubs': False,
        'nine_clubs': False,
        'eight_clubs': False,
        'seven_clubs': False,
        'six_clubs': False,
        'five_clubs': False,
        'four_clubs': False,
        'three_clubs': False,
        'two_clubs': False,
        'ace_diamonds': False,  # Diamonds
        'king_diamonds': False,
        'queen_diamonds': False,
        'jack_diamonds': False,
        'ten_diamonds': False,
        'nine_diamonds': False,
        'eight_diamonds': False,
        'seven_diamonds': False,
        'six_diamonds': False,
        'five_diamonds': False,
        'four_diamonds': False,
        'three_diamonds': False,
        'two_diamonds': False,
        'ace_hearts': False,  # Hearts
        'king_hearts': False,
        'queen_hearts': False,
        'jack_hearts': False,
        'ten_hearts': False,
        'nine_hearts': False,
        'eight_hearts': False,
        'seven_hearts': False,
        'six_hearts': False,
        'five_hearts': False,
        'four_hearts': False,
        'three_hearts': False,
        'two_hearts': False,
        'ace_spades': False,  # Spades
        'king_spades': False,
        'queen_spades': False,
        'jack_spades': False,
        'ten_spades': False,
        'nine_spades': False,
        'eight_spades': False,
        'seven_spades': False,
        'six_spades': False,
        'five_spades': False,
        'four_spades': False,
        'three_spades': False,
        'two_spades': False,
    }
    suits = {
        'clubs': False,  # 0
        'diamonds': False,  # 1
        'hearts': False,  # 2
        'spades': False,  # 3
    }
    for card in cards_list:
        card_name = get_card_name(card)
        if card_name:
            cards[card_name] = True
    suits['clubs'] = any([card.suit == 0 for card in cards_list])
    suits['diamonds'] = any([card.suit == 1 for card in cards_list])
    suits['hearts'] = any([card.suit == 2 for card in cards_list])
    suits['spades'] = any([card.suit == 3 for card in cards_list])
    return cards, suits


class RewardFunction:
    """
    The reward function an agent optimizes to win at Hearts.

    Calling this returns the reward.
    """

    def __init__(self, env: HeartsEnv):
        self.env = env
        self.game = env.game

    def __call__(self, *args, **kwargs) -> Reward:
        return self.compute_reward(*args, **kwargs)

    def compute_reward(
            self,
            player_index: int,
            prev_active_player_index: int,
            trick_is_over: bool,
    ) -> Reward:
        """Return the reward for the player with the given index.

        It is important to keep in mind that most of the time, the
        arguments are unrelated to the player getting their reward. This
        is because agents receive their reward only when it is their
        next turn, not right after their turn. Due to this peculiarity,
        it is encouraged to use `self.game.prev_played_cards`,
        `self.game.prev_was_illegals`, and others.

        Args:
            player_index (int): Index of the player to return the reward
                for. This is most of the time _not_ the player that took
                the action (which is given by `prev_active_player_index`).
            prev_active_player_index (int): Index of the previously
                active player that took the action. In other words, the
                active player index before the action was taken.
            trick_is_over (bool): Whether the action ended the trick.

        Returns:
            Reward: Reward for the player with the given index.
        """
        if self.game.prev_was_illegals[player_index]:
            return -self.game.max_penalty * self.game.max_num_cards_on_hand

        card = self.game.prev_played_cards[player_index]

        card_in_hands = self.game.prev_hands[player_index]

        if card is None:
            # The agent did not take a turn until now; no information
            # to provide.
            return 0

        # Get card name and cards in hand and table
        card_name = get_card_name(card)
        table_cards, table_suits = check_relevant_cards(self.game.prev_table_cards)
        in_hand, in_hand_suits = check_relevant_cards(card_in_hands)

        # Check if ace or king of spades are in the table
        ace_or_king = table_cards['ace_spades'] or table_cards['king_spades']

        # If queen of spades was played
        if card_name == 'queen_spades':
            # Reward if the leading suit was not spades (got rid of it)
            if self.game.leading_suit != 3:
                return self.game.max_penalty * 2

            # Reward if the ace or king of spades were in the table
            if ace_or_king:
                return self.game.max_penalty * 2
        else:
            # Punish if the ace or king of spades were in the table or
            # spades was not the leading suit and the player has the queen
            # in hand

            if (ace_or_king or self.game.leading_suit != 3) and \
                    in_hand['queen_spades']:
                return -self.game.max_penalty / 2

        # If we player a low card on a trick with no penalty but could have used a higher card - punish
        # If we player a high card on a trick with no penalty but could have used a lower card - reward

        # If the trick had no penalty
        if self.game.prev_trick_penalty == 0:
            # If
            pass

        if trick_is_over:

            if self.game.has_shot_the_moon(player_index):
                return self.game.max_penalty * self.game.max_num_cards_on_hand

            # # If we won a trick with card of rank x but could have won it with a card of rank y > x - punish
            # # If we won a trick with card of rank x but could have won it with a card of rank y < x - reward

            # If we won the trick
            if self.game.prev_trick_winner_index == player_index:

                # Find highest card of the leading suit in the table
                highest_rank = 0
                for card_in_table in self.game.prev_table_cards:
                    if card_in_table.suit == self.game.leading_suit:
                        if card_in_table.rank > highest_rank:
                            highest_rank = card_in_table.rank

                # Find if we have either a lower or higher card that could have won
                has_higher = False
                has_lower = False
                for card_in_hand in card_in_hands:
                    if card.suit == card_in_hand.suit:

                        # Check if we have a higher card of that suit in hand
                        # We could have won by playing that
                        if card_in_hand.rank > card.rank:
                            has_higher = True

                        # Check if we have a lower card of that suit in hand, that
                        # is higher than the highest on the table
                        elif card_in_hand.rank < card.rank:
                            if card.rank > highest_rank:
                                has_lower = True

                if has_higher:
                    return -1 * len(self.game.prev_table_cards)   # Could have won with higher card, punish
                if has_lower:
                    return 1 * len(self.game.prev_table_cards)    # Could have won with lower card, reward

        # penalty = self.game.penalties[player_index]

        # if self.game.is_done():
        #     return -penalty

        if self.game.prev_trick_winner_index == player_index:
            if self.game.prev_trick_penalty == 0:
                return 5
            else:
                assert self.game.prev_trick_penalty is not None
                return -self.game.prev_trick_penalty
        return 1
        # return -penalty
