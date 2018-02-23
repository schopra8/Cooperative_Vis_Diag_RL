class QBot(object):
    """Abstracts a Q-Bot for asking questions about a photo
    """
    def encode_captions(self, captions):
        """Encodes captions.

        Args:
            captions: [Batch Size, Caption Encoding Dimensions]
        Returns:
            captions: encoded captions  
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def encode_facts(self, questions, answers):
        """Encodes questions and answers into a fact (Fact Encoder)

        Args:
            questions: questions asked by the Q-Bot in the most recent round [Batch Size, 1]
            answers: answers given by the A-Bot in response to the questions [Batch Size, 1]
        Returns:
            facts: encoded facts that combine the question and answer
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def encode_state_histories(self, prev_states, recent_facts):
        """Encodes states as a combination of facts for a given round (State/History Encoder)

        Args:
            prev_states: [(q_1, a_1, q_2, a_2) ... ] List of tuples
            recent_facts: [(q_n, a_n), ...] List of yuples
        Returns:
            state: encoded state that combines the current facts and previous facts [Batch Size, 1]
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def decode_questions(self, states):
        """Decodes (generates) questions given the current states (Question Decoder)

        Args:
            states (list of tuples) [Batch Size]: encoded states
        Returns:
            questions: (list of ints) [Batch Size]
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def generate_image_representations(self, states):
        """Guesses images given the current states (Feature Regression Network)

        Args:
            state: encoded states [Batch Size, 1]
        Returns:
            image_repr: representation of the predicted images [Batch Size, 1]
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def get_q_values(self, states):
        """Returns all Q-values for all actions given states

        Args:
            state: encoded states [Batch Size, 1]
        Returns:
            values: mapping of actions to expected return values [Batch Size, 1]
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def get_questions(self, states):
        """Returns questions according to some exploration policy given encoded states

        Args:
            state: encoded states [Batch Size, 1]
        Returns:
            questions: questions that Q Bot will ask [Batch Size, 1]
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")
