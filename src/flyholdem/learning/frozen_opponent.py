"""Frozen native snapshot opponent; one learning player per hand remains separate."""
import hashlib
import numpy as np
from flyholdem.interface.population import NeuralController
from flyholdem.interface.frozen import load_frozen
from flyholdem.poker.infoset import canonical_state
from flyholdem.poker.observation import canonical_bytes
from flyholdem.provenance import identity


class FrozenNeuralOpponent:
    def __init__(self,controller,model_sha256):
        if not isinstance(controller,NeuralController) or type(model_sha256) is not str or len(model_sha256)!=64:
            raise ValueError('A native controller and pinned snapshot identity are required')
        self.controller=controller;self.brain=controller.brain;self.model_sha256=model_sha256
        self.weights_sha256=hashlib.sha256(self.brain.weights.tobytes()).hexdigest()
        self.version='frozen-native-opponent-v1:'+model_sha256;self.name='frozen-native:'+model_sha256
        self.decisions=[]

    @classmethod
    def load(cls,model,graph,expected_sha256):
        return cls(load_frozen(model,graph,expected_sha256),expected_sha256)

    def _check_weights(self):
        if hashlib.sha256(self.brain.weights.tobytes()).hexdigest()!=self.weights_sha256:
            raise ValueError('Frozen snapshot opponent weights changed')

    def begin_hand(self):
        self._check_weights();self.brain.reset_dynamics();self.decisions=[]

    def act(self,observation):
        visible=canonical_state(observation);before=canonical_bytes(visible)
        decision=self.controller.decide(visible,temperature=0)
        scores=np.asarray(decision['scores']);legal=np.asarray(visible['legal_mask'],dtype=bool)
        if (canonical_bytes(visible)!=before or scores.shape!=(5,) or not np.isfinite(scores).all()
                or decision['decoder_fallback'] or decision['selected']!=int(np.argmax(np.where(legal,scores,-np.inf)))):
            raise ValueError('Frozen opponent must use its actual legal native argmax')
        self.controller.commit_interval()
        self.decisions.append({'observation_sha256':hashlib.sha256(before).hexdigest(),
            'selected':decision['selected'],'scores':decision['scores'],'legal_mask':decision['legal_mask'],
            'counts_sha256':hashlib.sha256(decision['counts'].tobytes()).hexdigest()})
        return decision['selected']

    def finish_hand(self):
        self._check_weights()
        return {'model_sha256':self.model_sha256,'weights_sha256':self.weights_sha256,'weights_unchanged':True,
            'learning':False,'teacher_connected':False,'decisions':self.decisions,
            'decisions_sha256':identity(self.decisions)}
