# Copyright (C) 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from ctypes import (CFUNCTYPE,
                    cdll,
                    c_bool, c_char_p, c_int, c_uint, c_void_p)

#from auth_helpers import CredentialsRefresher
from event import Event, IterableEventQueue

LISTENER = CFUNCTYPE(None, c_int, c_char_p)


class UnsupportedPlatformError(Exception):
    """Raised if the OS is unsupported by the Assistant."""
    pass


class Assistant(object):
    """Client for the Google Assistant Library.

    Provides basic control functionality and lifecycle handling for the Google
    Assistant. It is best practice to use the Assistant as a ContextManager:

        with Assistant(credentials) as assistant:

    This allows the underlying native implementation to properly handle memory
    management. Once started, the Assistant generates a stream of Events
    relaying the various states the Assistant is currently in, for example:

        ON_CONVERSATION_TURN_STARTED
        ON_END_OF_UTTERANCE
        ON_RECOGNIZING_SPEECH_FINISHED:
            {'text': 'what time is it'}
        ON_RESPONDING_STARTED:
            {'is_error_response': False}
        ON_RESPONDING_FINISHED
        ON_CONVERSATION_TURN_FINISHED:
            {'with_follow_on_turn': False}

    See google.assistant.event.EventType for details on all events and their
    arguments.

    Glossary:
        Hotword: The phrase the Assistant listens for when not muted:

            "OK Google" OR "Hey Google"

        Turn: A single user request followed by a response from the Assistant.

        Conversation: One or more turns which result in a desired final result
            from the Assistant:

            "What time is it?" -> "The time is 6:24 PM" OR
            "Set a timer" -> "Okay, for how long?" ->
            "5 minutes" -> "Sure, 5 minutes, starting now!"
    """

    def __init__(self, credentials):
        """Initializes a new Assistant with OAuth2 credentials.

        If the user has not yet logged into the Assistant, then a new
        authentication flow will be started asking the user to login. Once
        initialized, the Assistant will be ready to start (see self.start()).

        Args:
            credentials(google.oauth2.credentials.Credentials): The user's
                Google OAuth2 credentials.

        Raises:
            UnsupportedPlatformError: If the current processor/operating system
                is not supported by the Google Assistant.
        """
        self._event_queue = IterableEventQueue()
        self._load_lib()
        self._credentials_refresher = None

        self._event_callback = LISTENER(self)
        self._inst = c_void_p(
            self._lib.assistant_new(self._event_callback))

        # self._credentials_refresher = CredentialsRefresher(
        #     credentials, self._set_credentials)
        # self._credentials_refresher.start()

    def __enter__(self):
        """Returns self."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Frees allocated memory belonging to the Assistant."""
        if self._credentials_refresher:
            self._credentials_refresher.stop()
            self._credentials_refresher = None
        self._lib.assistant_free(self._inst)

    def __call__(self, event_type, event_data):
        """Adds a new event to the event queue returned from start().

        Args:
            event_type(int): A numeric id corresponding to an event in
                google.assistant.event.EventType.
            event_data(str): A serialized JSON string with key/value pairs
                for event arguments.
        """
        self._event_queue.offer(Event(event_type, event_data))

    def start(self):
        """Starts the Assistant, which includes listening for a hotword.

        Once start() is called, the Assistant will begin processing data from
        the 'default' ALSA audio source, listening for the hotword. This will
        also start other services provided by the Assistant, such as
        timers/alarms. This method can only be called once. Once called, the
        Assistant will continue to run until __exit__ is called.

        Returns:
            google.assistant.event.IterableEventQueue: A queue of events
                that notify of changes to the Assistant state.
        """
        self._lib.assistant_start(self._inst)
        return self._event_queue

    def set_mic_mute(self, is_muted):
        """Stops the Assistant from listening for the hotword.

        Allows for disabling the Assistant from listening for the hotword.
        This provides functionality similar to the privacy button on the back
        of Google Home.

        This method is a no-op if the Assistant has not yet been started.

        Args:
            is_muted(bool): True stops the Assistant from listening and False
                allows it to start again.
        """
        self._lib.assistant_set_mic_mute(self._inst, is_muted)

    def start_conversation(self):
        """Manually starts a new conversation with the Assistant.

        Starts both recording the user's speech and sending it to Google,
        similar to what happens when the Assistant hears the hotword.

        This method is a no-op if the Assistant is not started or has been
        muted.
        """
        self._lib.assistant_start_conversation(self._inst)

    def stop_conversation(self):
        """Stops any active conversation with the Assistant.

        The Assistant could be listening to the user's query OR responding. If
        there is no active conversation, this is a no-op.
        """
        self._lib.assistant_stop_conversation(self._inst)

    def _set_credentials(self, credentials):
        """Sets Google account OAuth2 credentials for the current user.

        Args:
            credentials(google.oauth2.credentials.Credentials): OAuth2
                Google account credentials for the current user.
        """
        # The access_token should always be made up of only ASCII
        # characters so this encoding should never fail.
        access_token = credentials.token.encode('ascii')
        self._lib.assistant_set_access_token(self._inst,
                                             access_token, len(access_token))

    def _load_lib(self):
        """Dynamically loads the Google Assistant Library.

        Automatically selects the correct shared library for the current
        platform and sets up bindings to its C interface.

        Raises:
            UnsupportedPlatformError: If the current processor or OS
                is not supported by the Google Assistant.
        """
        os_name = os.uname()[0]
        platform = os.uname()[4]

        lib_name = 'libassistant_embedder_' + platform + '.so'
        lib_path = os.path.join(os.path.dirname(__file__), lib_name)

        if os_name != 'Linux' or not os.path.isfile(lib_path):
            raise UnsupportedPlatformError(platform + ' is not supported.')

        self._lib = cdll.LoadLibrary(lib_path)

        # void* assistant_new(EventCallback listener);
        self._lib.assistant_new.arg_types = [LISTENER]
        self._lib.assistant_new.restype = c_void_p

        # void assistant_free(void* instance);
        self._lib.assistant_free.argtypes = [c_void_p]
        self._lib.assistant_free.restype = None

        # void assistant_start(void* assistant);
        self._lib.assistant_start.arg_types = [c_void_p]
        self._lib.assistant_start.res_type = None

        # void assistant_set_access_token(
        #     void* assistant, const char* access_token, size_t length);
        self._lib.assistant_set_access_token.arg_types = [
            c_void_p, c_char_p, c_uint
        ]
        self._lib.assistant_set_access_token.res_type = None

        # void assistant_set_mic_mute(void* assistant, bool is_muted);
        self._lib.assistant_set_mic_mute.arg_types = [c_void_p, c_bool]
        self._lib.assistant_set_mic_mute.res_type = None

        # void assistant_start_conversation(void* assistant);
        self._lib.assistant_start_conversation.arg_types = [c_void_p]
        self._lib.assistant_start_conversation.res_type = None

        # void assistant_stop_conversation(void* assistant);
        self._lib.assistant_stop_conversation.arg_types = [c_void_p]
        self._lib.assistant_stop_conversation.res_type = None
