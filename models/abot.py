class ABot(object):
    """Abstracts an A-Bot for answering questions about a photo
    """
    def encode_question(self, question):
        """Encodes a question given the current state (Question Encoder)

        Args:
            question: A question that the Q-Bot asked
        Returns:
            question_encoding: An encoding of the question
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def encode_fact(self, question, answer):
        """Encodes a fact as a combination of a question and answer (Fact Encoder)

        Args:
            question: A question asked by the Q-Bot in the most recent round
            answer: The answer given by the A-Bot in response to the question
        Returns:
            fact: An encoded fact that combines the question and answer
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def encode_state_history(self, question_encoding, fact):
        """Encodes a state as a combination of facts (State/History Encoder)

        Args:
            question_encoding: An encoded question
            fact: An encoded fact
        Returns:
            state: An encoded state that combines question_encodings and facts
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def decode_answer(self, state):
        """Generates an answer given the current state (Answer Decoder)

        Args:
            state: An encoded state
        Returns:
            answer: An answer
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def get_q_values(self, state):
        """Returns all Q-values for all actions given state

        Args:
            state: An encoded state
        Returns:
            values: A representation of action to expected value
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def get_action(self, state):
        """Returns an action according to some exploration policy given an encoded state

        Args:
            state: An encoded state
        Returns:
            action: A question for the Q-Bot to ask
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")
