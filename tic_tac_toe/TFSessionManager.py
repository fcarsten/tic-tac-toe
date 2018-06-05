import tensorflow as tf


class TFSessionManager:
    """
    Class to manage the TensorFlow session for all NN based players.

    Since it seems best to just have a single TensorFlow session which is shared by all
    players, this class is used to manage that one sessions.

    In order for this to work, all players need to be careful to put their graph into a unique name scope.
    """

    session = None

    @classmethod
    def get_session(cls) -> tf.Session:
        """
        Returns the one shared session between all players. If no session has been explicitly set, a new session
        is created.
        :return: The one shared session between all players.
        """
        if TFSessionManager.session is None:
            TFSessionManager.session = tf.Session()
        return TFSessionManager.session

    @classmethod
    def set_session(cls, session: tf.Session):
        """
        Explicitly sets the input parameter session as the one session to be shared by all instances.
        If a session already was set previously and it is not identical with the new one, the old session
        will be explicitly closed before setting the new one.
        :param session:
        :return:
        """
        if TFSessionManager.session is not None and TFSessionManager.session != session:
            TFSessionManager.session.close()

        TFSessionManager.session = session
