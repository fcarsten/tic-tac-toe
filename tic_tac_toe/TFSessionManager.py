import tensorflow as tf


class TFSessionManager:

    session = None

    @classmethod
    def get_session(cls) -> tf.Session:
        if TFSessionManager.session is None:
            TFSessionManager.session = tf.Session()
        return TFSessionManager.session

    @classmethod
    def set_session(cls, session : tf.Session):
        if TFSessionManager.session is not None and TFSessionManager.session != session:
            TFSessionManager.session.close()

        TFSessionManager.session = session
