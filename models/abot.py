class ABot(object):
    """Abstracts an A-Bot for answering questions about a photo
    """
    def encode_images_captions(self, captions, images):
        """Encodes the captions and the images into the states

        Args:
            captions [Batch Size, Caption] : Gives the captions for the current round of dialog
            images [Batch Size, Image] : The images for the dialog
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def encode_questions(self, questions):
        """Encodes questions (Question Encoder)

        Args:
            questions: questions that the Q-Bot asked [Batch Size, 1]
        Returns:
            question_encodings: encoding of the questions [Batch Size,]
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def encode_facts(self, questions, answers):
        """Encodes questions and answers into a fact (Fact Encoder)

        Args:
            questions: questions asked by the Q-Bot in the most recent round [Batch Size]
            answers: answers given by the A-Bot in response to the questions [Batch Size]
        Returns:
            facts: encoded facts that combine the question and answer
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def encode_state_histories(self, images, captions, question_encodings, recent_facts, prev_states):
        """Encodes states as a combination of facts (State/History Encoder)

        Args:
            images [Batch Size, Image] : The images for the dialog
            captions [Batch Size, Caption] : Gives the captions for the current round of dialog
            question_encodings: encoded questions [Batch Size]
            recent_facts: encoded facts [Batch Size]
            prev_states: prev states [Batch Size]
        Returns:
            states: encoded states that combine images, captions, question_encodings, recent_facts
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def decode_answers(self, states):
        """Generates answer given the current states (Answer Decoder)

        Args:
            states: encoded states
        Returns:
            answers: answers
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def generate_image_representations(self, states):
        """Guesses images given the current states (Feature Regression Network)

        Args:
            state: encoded states [Batch Size, 1]
        Returns:
            image_repr: representation of the predicted images [Batch Size, 1]
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def get_answers(self, states):
        """Returns answers according to some exploration policy given encoded states

        Args:
            state: encoded states [Batch Size, 1]
        Returns:
            answers: answers that A Bot will provide [Batch Size, 1]
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")