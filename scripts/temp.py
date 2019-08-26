
class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode                                = 0
    global_moving_average_reward                  = 0
    best_score                                    = 0
    save_lock                                     = threading.Lock()

    def __init__(self,
                state_size,
                action_size,
                global_model,
                opt,
                result_queue,
                idx,
                game_name='CartPole-v0',
                save_dir='/tmp'):
        super(Worker, self).__init__()
        self.state_size                           = state_size
        self.action_size                          = action_size
        self.result_queue                         = result_queue
        self.global_model                         = global_model
        self.opt                                  = opt
        self.local_model                          = A3C(self.state_size, self.action_size)
        self.worker_idx                           = idx
        self.game_name                            = game_name
        self.env                                  = gym.make(self.game_name).unwrapped
        self.save_dir                             = save_dir
        self.epi_loss                              = 0.0

    def run(self):
        total_step                                = 1
        mem                                       = Memory()
        while Worker.global_episode < args.max_eps:
            current_state                         = self.env.reset()
            mem.clear()
            epi_reward                             = 0.
            epi_steps                              = 0
            self.epi_loss                          = 0

            time_count                            = 0
            done                                  = False
            while not done:
                # Boltzmann action selection
                logits, _                         = self.local_model(tf.convert_to_tensor(current_state[None, :],dtype=tf.float32))
                probs                             = tf.nn.softmax(logits)
                action                            = np.random.choice(self.action_size, p=probs.numpy()[0])
                # Play action
                new_state, reward, done, _        = self.env.step(action)
                if done:
                    adjusted_reward               = -1
                epi_reward                         = epi_reward + reward
                # Update memory
                mem.add(current_state, action, reward)


####################################################################################################
                if ((time_count % args.update_freq)==0) or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss                = self.compute_loss(done, new_state, mem, args.gamma)
                    self.epi_loss                  = self.epi_loss + total_loss
                    # Calculate local gradients
                    grads                         = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())
                    mem.clear()
                    time_count                    = 0
                    # Print results at end of game
                    if done:
                        
                        Worker.global_moving_average_reward= \
                            record(Worker.global_episode, epi_reward, adjusted_reward, self.worker_idx,
                                Worker.global_moving_average_reward, self.result_queue,
                                self.epi_loss, epi_steps)
                        # Lock to save model and prevent data races
                        if epi_reward > Worker.best_score:
                            with Worker.save_lock:
                                print('Saving best model to {}, '
                                            'episode score: {}'.format(self.save_dir, epi_reward))
                                self.global_model.save_weights(os.path.join(self.save_dir,'model_{}.h5'.format(self.game_name)))
                                Worker.best_score = epi_reward
                        Worker.global_episode     = Worker.global_episode + 1
####################################################################################################
                epi_steps                          = epi_steps + 1
                time_count                        = time_count + 1
                current_state                     = new_state
                total_step                        = total_step + 1
        self.result_queue.put(None)

    def compute_loss(self,done,new_state,memory,gamma=0.99):
        if done:
            reward_sum                            = 0.
        else:
            reward_sum                            = self.local_model(tf.convert_to_tensor(new_state[None, :],dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards                        = []
        for reward in memory.rewards[::-1]: # reverse
            reward_sum                            = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values                            = self.local_model(tf.convert_to_tensor(np.vstack(memory.states),dtype=tf.float32))
        # Get our advantages
        advantage                                 = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                                        dtype=tf.float32) - values
        # Value loss
        value_loss                                = advantage ** 2
        # Calculate our policy loss
        policy                                    = tf.nn.softmax(logits)
        entropy                                   = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy,logits=logits)
        policy_loss                               = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,logits=logits)
        policy_loss                               = policy_loss * tf.stop_gradient(advantage)
        policy_loss                               = policy_loss - 0.01 * entropy
        total_loss                                = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss



import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "ylee01603@gmail.com"  # Enter your address
receiver_email = "young.lee@hotmail.co.uk"  # Enter receiver address
display_name = 'Young Lee'
password = 'lol54321'#input("Type your password and press enter: ")
message = '''
Subject: Hi there
This message is sent from Python.'''

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(user=sender_email, password=password)
    server.sendmail(sender_email, receiver_email, message)


import email
import smtplib

def send_email(sender_email, to_email, password, subject_string, message_string):
    msg = email.message_from_string(message_string)
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject_string
    s = smtplib.SMTP("smtp.live.com",587)
    s.ehlo() 
    s.starttls()
    s.ehlo()
    s.login(sender_email, password)
    s.sendmail(sender_email, to_email, msg.as_string())
    s.quit()
    print('Email sent.')

send_email(sender_email='young.lee@hotmail.co.uk', to_email='young.lee@hotmail.co.uk', 
           password='Two0whee!2', subject_string='Checkpoint for Python job', 
           message_string='hi')

import time
(int(time.time()*100) % 1999)**2 % 199999
import matplotlib.pyplot as plt

env = gym.make('DemonAttackNoFrameskip-v4').unwrapped
import numpy as np
import random

# env.seed(711110)
# np.random.seed(10)
env._seed(70)
_ = env.reset()
for i in range(500):
    _ = env.step(np.random.choice(range(6),1)[0])[0]

plt.imshow(env.step(3)[0])
plt.show()
from scipy.special import softmax
softmax([-1,-1,-0.5,1,1,1])