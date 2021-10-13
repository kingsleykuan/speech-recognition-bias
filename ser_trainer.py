import torch
from sklearn.metrics import classification_report, f1_score

from trainer import Trainer
from utils import recursive_to_device


class SpeechEmotionRecognitionTrainer(Trainer):
    def __init__(
            self,
            data_loader_train,
            data_loader_eval,
            model,
            num_epochs=100,
            steps_per_log=1,
            epochs_per_eval=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=0,
            log_dir=None,
            save_path=None,
            use_ratings=False):
        super().__init__(
            data_loader_train,
            data_loader_eval,
            model,
            num_epochs=num_epochs,
            steps_per_log=steps_per_log,
            epochs_per_eval=epochs_per_eval,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            log_dir=log_dir,
            save_path=save_path)
        self.use_ratings = use_ratings
        self.current_f1_score = float('-inf')
        self.best_f1_score = float('-inf')
        self.best_epoch = None

    def train_forward(self, data):
        data = recursive_to_device(data, self.device, non_blocking=True)

        log_mel_spec = data['log_mel_spec']

        if self.use_ratings:
            emotion_label = data['emotion_rating_labels']
        else:
            emotion_label = data['emotion_label']

        outputs = self.model(log_mel_spec, emotion_label)
        loss = outputs['loss']
        return outputs, loss

    def train_log(self, outputs, loss):
        if self.writer is not None:
            self.writer.add_scalar('loss', loss, global_step=self.global_step)

    def eval_forward(self, data):
        data = recursive_to_device(data, self.device, non_blocking=True)

        log_mel_spec = data['log_mel_spec']

        if self.use_ratings:
            emotion_label = data['emotion_rating_labels']
        else:
            emotion_label = data['emotion_label']

        outputs = self.model(log_mel_spec)

        outputs = torch.round(torch.sigmoid(outputs['logits']))
        labels = torch.round(emotion_label)
        return outputs, labels

    def eval_log(self, outputs, labels):
        outputs = torch.cat(outputs).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()

        self.current_f1_score = f1_score(
            labels, outputs, average='macro', zero_division=0)

        if self.current_f1_score >= self.best_f1_score:
            self.best_f1_score = self.current_f1_score
            self.best_epoch = self.epoch

        if self.writer is not None:
            self.writer.add_scalar(
                'f1_score', self.current_f1_score, global_step=self.epoch)

        print(classification_report(
            labels,
            outputs,
            target_names=self.data_loader_eval.dataset.emotions,
            digits=5,
            zero_division=0))

    def save_model(self):
        if self.save_path is not None and \
                self.current_f1_score >= self.best_f1_score:
            self.model.save(
                self.save_path,
                epoch=self.best_epoch,
                f1_score=self.best_f1_score)
