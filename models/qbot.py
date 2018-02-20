class QBot(object):
    """Abstracts a Q-Bot for asking questions about a photo
    """
    def encode_fact(self, question, answer):
        """Encodes a question and answer into a fact (Fact Encoder)

        Args:
            question: A question asked by the Q-Bot in the most recent round
            answer: The answer given by the A-Bot in response to the question
        Returns:
            fact: An encoded fact that combines the question and answer
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def encode_state_history(self, fact):
        """Encodes the state as a combination of facts (State/History Encoder)

        Args:
            fact: An encoded fact
        Returns:
            state: An encoded state that combines the current fact and previous facts
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def decode_question(self, state):
        """Decodes (generates) a question given the current state (Question Decoder)

        Args:
            state: An encoded state
        Returns:
            question: A question that the Q-Bot will ask
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def generate_image_representation(self, state):
        """Guesses an image given the current state (Feature Regression Network)

        Args:
            state: An encoded state
        Returns:
            image_repr: A representation of the predicted image
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def get_q_values(self, state):
        """Returns all Q-values for all actions given state

        Args:
            state: An encoded state
        Returns:
            values: A representation of action to expected value
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def get_action(self, state):
        """Returns an action according to some exploration policy given an encoded state

        Args:
            state: An encoded state
        Returns:
            action: A question for the Q-Bot to ask
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")
