import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

# Acoustic Without GRU, which is essentially a Concurrent CNN
class AcousticWithoutGRU(nn.Module):
    def __init__(self, num_classes, midfeat, momentum):
        super(AcousticWithoutGRU, self).__init__()
        inp_dim = 1
        hidden1 = 48
        hidden2 = 64
        hidden3 = 96
        out_dim = 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_dim, out_channels=hidden1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden1,momentum),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden1, out_channels=hidden1, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden1,momentum),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden2,momentum),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden2, out_channels=hidden2, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden2,momentum),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden2, out_channels=hidden3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden3,momentum),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden3, out_channels=hidden3, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden3,momentum),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden3, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim,momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim,momentum=momentum),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2))
        )
        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)
        self.linear = nn.Linear(768, 512, bias=True)
        self.fc = nn.Linear(512, num_classes, bias=True)
        self.dp = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.dp1 = nn.Dropout(p=0.5, inplace=False)
        self.sg = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dp(self.conv1(x))
        x = self.dp(self.conv2(x))
        x = self.dp(self.conv3(x))
        x = self.dp(self.conv4(x))
        x = x.transpose(1,2).flatten(2)
        x = self.fc5(x)
        x = x.transpose(1,2)
        x = self.bn5(x)
        x = x.transpose(1,2)
        x = self.relu(x)
        x = self.dp1(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dp1(x)
        output = self.sg(self.fc(x))
        return output


# Acoustic Model
class AcousticModel(nn.Module):
    def __init__(self, num_classes, midfeat, momentum):
        super(AcousticModel, self).__init__()
        inp_dim = 1
        hidden1 = 48
        hidden2 = 64
        hidden3 = 96
        out_dim = 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_dim, out_channels=hidden1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden1,momentum),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden1, out_channels=hidden1, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden1,momentum),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden2,momentum),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden2, out_channels=hidden2, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden2,momentum),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden2, out_channels=hidden3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden3,momentum),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden3, out_channels=hidden3, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden3,momentum),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden3, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim,momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim,momentum=momentum),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2))
        )
        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)
        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.fc = nn.Linear(512, num_classes, bias=True)
        self.dp = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.dp1 = nn.Dropout(p=0.5, inplace=False)
        self.sg = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dp(self.conv1(x))
        x = self.dp(self.conv2(x))
        x = self.dp(self.conv3(x))
        x = self.dp(self.conv4(x))
        x = x.transpose(1,2).flatten(2)
        x = self.fc5(x)
        x = x.transpose(1,2)
        x = self.bn5(x)
        x = x.transpose(1,2)
        x = self.relu(x)
        x = self.dp1(x)
        (x, _) = self.gru(x)
        x = self.dp1(x)
        output = self.sg(self.fc(x))
        return output

# A Simple Neural Network
class Net(nn.Module):
    def __init__(self, num_classes, midfeat, momentum):
        super(Net, self).__init__()
        n_out = num_classes
        self.fc1 = nn.Linear(229, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, 100)
        self.fc4 = nn.Linear(100, n_out)

    def forward(self, x):
        drop_p = 0.2
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = F.sigmoid(self.fc4(x4))
        return x5
    
# Overall Final Model which uses Acoustic Models and Conditions their output on features
class CRNN_Conditioning(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(CRNN_Conditioning, self).__init__()
        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate//frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate//2
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        midfeat = 1792
        momentum = 0.01
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
            hop_length=hop_size, win_length=window_size, window=window, 
            center=center, pad_mode=pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
            n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
            amin=amin, top_db=top_db, freeze_parameters=True)
        
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.frame_model =      AcousticModel(classes_num, midfeat, momentum)
        self.reg_onset_model =  AcousticModel(classes_num, midfeat, momentum)
        self.reg_offset_model = AcousticModel(classes_num, midfeat, momentum)
        self.velocity_model =   AcousticModel(classes_num, midfeat, momentum)
        self.reg_onset_gru = nn.GRU(input_size=88 * 2, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)
        self.frame_gru = nn.GRU(input_size=88 * 3, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)

    def forward(self, input):
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        frame_output = self.frame_model(x)  # (batch_size, time_steps, classes_num)
        reg_onset_output = self.reg_onset_model(x)  # (batch_size, time_steps, classes_num)
        reg_offset_output = self.reg_offset_model(x)    # (batch_size, time_steps, classes_num)
        velocity_output = self.velocity_model(x)    # (batch_size, time_steps, classes_num)
 
        # Use velocities to condition onset regression
        x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach()), dim=2)
        (x, _) = self.reg_onset_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))
        """(batch_size, time_steps, classes_num)"""

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        frame_output = torch.sigmoid(self.frame_fc(x))  # (batch_size, time_steps, classes_num)
        """(batch_size, time_steps, classes_num)"""

        output_dict = {
            'reg_onset_output': reg_onset_output, 
            'reg_offset_output': reg_offset_output, 
            'frame_output': frame_output, 
            'velocity_output': velocity_output}

        return output_dict

# CRNN Model (uses GRU)
class CRNN(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(CRNN, self).__init__()
        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate // 2
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        midfeat = 1792
        momentum = 0.01
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
            hop_length=hop_size, win_length=window_size, window=window, 
            center=center, pad_mode=pad_mode, freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
            n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
            amin=amin, top_db=top_db, freeze_parameters=True)
        
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.frame_model =      AcousticModel(classes_num, midfeat, momentum)
        self.reg_onset_model =  AcousticModel(classes_num, midfeat, momentum)
        self.reg_offset_model = AcousticModel(classes_num, midfeat, momentum)
        self.velocity_model =   AcousticModel(classes_num, midfeat, momentum)
        self.reg_onset_gru = nn.GRU(input_size=88 * 2, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)
        self.frame_gru = nn.GRU(input_size=88 * 3, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)
    def forward(self, input):
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        frame_output = self.frame_model(x)  # (batch_size, time_steps, classes_num)
        reg_onset_output = self.reg_onset_model(x)  # (batch_size, time_steps, classes_num)
        reg_offset_output = self.reg_offset_model(x)    # (batch_size, time_steps, classes_num)
        velocity_output = self.velocity_model(x)    # (batch_size, time_steps, classes_num)

        output_dict = {
            'reg_onset_output': reg_onset_output, 
            'reg_offset_output': reg_offset_output, 
            'frame_output': frame_output, 
            'velocity_output': velocity_output}
        return output_dict
    
# Concurrent Convolutional Neural Network
class CCNN(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(CCNN, self).__init__()
        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate // 2
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        midfeat = 1792
        momentum = 0.01
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
            hop_length=hop_size, win_length=window_size, window=window, 
            center=center, pad_mode=pad_mode, freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
            n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
            amin=amin, top_db=top_db, freeze_parameters=True)
        
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.frame_model =      AcousticWithoutGRU(classes_num, midfeat, momentum)
        self.reg_onset_model =  AcousticWithoutGRU(classes_num, midfeat, momentum)
        self.reg_offset_model = AcousticWithoutGRU(classes_num, midfeat, momentum)
        self.velocity_model =   AcousticWithoutGRU(classes_num, midfeat, momentum)
        self.reg_onset_gru = nn.GRU(input_size=88 * 2, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)
        self.frame_gru = nn.GRU(input_size=88 * 3, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)
    def forward(self, input):
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        frame_output = self.frame_model(x)  # (batch_size, time_steps, classes_num)
        reg_onset_output = self.reg_onset_model(x)  # (batch_size, time_steps, classes_num)
        reg_offset_output = self.reg_offset_model(x)    # (batch_size, time_steps, classes_num)
        velocity_output = self.velocity_model(x)    # (batch_size, time_steps, classes_num)
        output_dict = {
            'reg_onset_output': reg_onset_output, 
            'reg_offset_output': reg_offset_output, 
            'frame_output': frame_output, 
            'velocity_output': velocity_output}
        return output_dict
    
    

