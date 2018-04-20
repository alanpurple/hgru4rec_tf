import tensorflow as tf
import pandas as pd
import numpy as np
import json
from os import path

class HGRU4Rec(object):
    def __init__(self,s_size,u_size,batch_size,
            s_dropout,u_dropout,len_items,s_depth,u_depth):
        self.s_size=s_size
        self.u_size=u_size
        self.batch_size=batch_size
        self.s_dropout=s_dropout
        self.u_dropout=u_dropout
        self.item_len=len_items
        self.s_depth=s_depth
        self.u_depth=u_depth

    def top1_loss(self,logits):
        logits=tf.transpose(logits)
        total_loss=tf.reduce_mean(tf.sigmoid(logits-tf.diag_part(logits))+tf.sigmoid(logits**2),axis=0)
        answer_loss=tf.sigmoid(tf.diag_part(logits)**2)/self.batch_size
        return tf.reduce_mean(total_loss-answer_loss)

    # flag size - batch_size
    # flag meaning - 0: same session, 1: new session, 2: new user
    # mode - 0: train, 1: eval, 2: predict
    # return 
    def build_model(self,mode,step):
        self.s_state=[tf.placeholder(tf.float32,(self.batch_size,self.s_size),'Session_State') for _ in range(self.s_depth)]
        self.u_state=[tf.placeholder(tf.float32,(self.batch_size,self.u_size),'User_State') for _ in range(self.u_depth)]
        self.flags=tf.placeholder(tf.bool,self.batch_size,'Session_Reset_flags')
        self.X=tf.placeholder(tf.int32,[self.batch_size],name='input')
        if mode==0 or mode==1:
            self.Y=tf.placeholder(tf.int32,[self.batch_size],name='output')

        with tf.variable_scope('Embedding',reuse=tf.AUTO_REUSE):
            embedding=tf.get_variable('embedding',[self.item_len,self.s_size])
            w_embedding=tf.get_variable('w_embedding',[self.item_len,self.s_size])
            b_embedding=tf.get_variable('b_embedding',[self.item_len])

            x_embedded=tf.nn.embedding_lookup(embedding,self.X)
        # s_gru=tf.nn.rnn_cell.GRUCell(self.s_size,name='Session_GRU',reuse=tf.AUTO_REUSE)
        # apply dropout
        # s_gru=tf.nn.rnn_cell.DropoutWrapper(s_gru,self.s_dropout)
        s_cells=[tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.s_size,name='Session_GRU_{}'.format(i),reuse=tf.AUTO_REUSE),
                self.s_dropout) for i in range(self.s_depth)]
        s_gru=tf.nn.rnn_cell.MultiRNNCell(s_cells)
        
        # u_gru=tf.nn.rnn_cell.GRUCell(self.u_size,name='User_GRU', reuse=tf.AUTO_REUSE)
        # apply dropout
        # u_gru=tf.nn.rnn_cell.DropoutWrapper(u_gru,self.u_dropout)
        u_cells=[tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.u_size,name='User_GRU_{}'.format(i), reuse=tf.AUTO_REUSE),
                self.u_dropout) for i in range(self.u_depth)]
        u_gru=tf.nn.rnn_cell.MultiRNNCell(u_cells)

        with tf.name_scope('Session_Reset'):
            with tf.variable_scope('Session_Reset',reuse=tf.AUTO_REUSE):
                u_output,updated_state=u_gru(tf.layers.dense(self.s_state[-1],self.u_size),tuple(self.u_state))
                u_state_T = tf.where(self.flags,tf.transpose(updated_state,perm=(1,0,2)),tf.transpose(self.u_state,perm=(1,0,2)),name='user_update')
                u_state=tf.transpose(u_state_T,perm=(1,0,2))
                new_s_init_T=tf.transpose([tf.layers.dense(u_output,self.s_size)]*self.s_depth,perm=(1,0,2))
                s_gru_input_T=tf.where(self.flags,new_s_init_T,tf.transpose(self.s_state,perm=(1,0,2)),name='input_state_for_s_gru')
                s_gru_input=tf.transpose(s_gru_input_T,perm=(1,0,2))
                s_gru_input=tf.unstack(s_gru_input)
                
        
        output,s_final_state=s_gru(x_embedded,tuple(s_gru_input))                                                                                                                                                                                                                                                                                                                                                                                                         

        # train or eval
        if mode==0 or mode==1:
            with tf.name_scope('Loss_calculation'):
                sampled_w=tf.nn.embedding_lookup(w_embedding,self.Y)
                sampled_b=tf.nn.embedding_lookup(b_embedding,self.Y)
                logits=tf.matmul(output,sampled_w,transpose_b=True)+sampled_b
                logits=tf.tanh(logits)
                loss=self.top1_loss(logits)
                tf.summary.scalar('loss over batch',loss)
                self.merged=tf.summary.merge_all()
            if mode==0:
                # Can be other optimizer
                with tf.name_scope('optimize'):
                    optimizer=tf.train.AdamOptimizer(name='Final_Adam')
                    tvars=tf.trainable_variables()
                    train_op=optimizer.minimize(loss,step,tvars)
                    return s_final_state,loss,train_op,u_state
            else:
                return s_final_state,loss,u_state
        # predict
        else:
            logits=tf.matmul(output,w_embedding,transpose_b=True)+b_embedding
            logits=tf.tanh(logits)
            predictions=tf.diag_part(logits)
            return predictions

    def train(self,inputs,outputs,flags,num_epochs,model_name):
        print('start training')
        with tf.Graph().as_default():
            global_step=tf.train.get_or_create_global_step()
            s_final,loss,train_op,u_final = self.build_model(0,global_step)
            train_writer=tf.summary.FileWriter('./model')
            summary=tf.summary.merge_all()
            fetches=[s_final, loss, train_op,u_final,summary]
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                for n in range(num_epochs):
                    loss_sum=0
                    current_u_state=[np.zeros([self.batch_size,self.u_size],dtype=np.float32)]*self.u_depth
                    current_s_state=[np.zeros([self.batch_size,self.s_size],dtype=np.float32)]*self.s_depth
                    for i in range(len(inputs)):
                        current_flags=[elem==1 for elem in flags[i]]
                        for j in range(self.batch_size):
                            if flags[i][j]==2:
                                for k in range(self.s_depth):
                                    current_s_state[k][j]=0
                                for k in range(self.u_depth):
                                    current_u_state[k][j]=0

                        print('start building batch {}'.format(i))

                        print('Run session {}'.format(i))
                        feeds={
                                self.X:inputs[i],
                                self.Y:outputs[i],
                                self.flags:current_flags
                            }
                        for j in range(self.u_depth):
                            feeds[self.u_state[j]]=current_u_state[j]
                        for j in range(self.s_depth):
                            feeds[self.s_state[j]]=current_s_state[j]
                        result=sess.run(fetches,feeds)
                        current_u_state=result[3]
                        current_s_state=result[0]
                        # need to handle loss ( log, etc )
                        print('finishing batch: {}'.format(i))
                        train_writer.add_graph(sess.graph,i)
                        train_writer.add_summary(result[4],global_step=i)
                        loss_sum+=result[1]
                    print('loss for epoch {}: {}'.format(n+1,loss_sum))
                tf.train.Saver().save(sess,'./model/'+model_name+'.ckpt',global_step=n)
                    

# batch_size should be same or smaller than num_of_users
def make_train_eval_batch_data(filename,batch_size):
    df=pd.read_csv(filename+'.csv')
    # Make item dictionary
    item_dict=[]
    if not path.exists(filename+'.json'):
        item_dict=np.unique(df['Item Id']).tolist()
        #save dictionary with (same name).json
        with open(filename+'.json','w') as dict_file:
            json.dump(item_dict,dict_file)
    else:
        with open(filename+'.json','r') as dict_file:
            item_dict = json.load(dict_file)
    
    user_dict=df['User Id'].unique().tolist()
    num_users=len(user_dict)
    assert batch_size <= num_users
    alldata=[[] for _ in range(num_users)]
    current_session=[0]*num_users
    current_user_index=-1

    templist=[[] for _ in range(num_users)]
    for index,row in df.iterrows():
        user_index=user_dict.index(row['User Id'])
        if current_user_index==-1:
            current_user_index=user_index
        session_id=row['Session Id']
        item=item_dict.index(row['Item Id'])
        if current_session[user_index]==0:
            current_session[user_index]=session_id
        # new user
        if user_index!=current_user_index:
            alldata[current_user_index].append(templist[current_user_index])
            current_user_index=user_index
            current_session[user_index]=session_id
            templist[user_index]=[item]
        # same user new session
        elif current_session[user_index]!=session_id:
            alldata[user_index].append(templist[user_index])
            templist[user_index]=[item]
            current_session[user_index]=session_id
        else:
            templist[user_index].append(item)
        # End of data
        if index==len(df)-1:
            alldata[user_index].append(templist[user_index])

    

    inputs=[]
    # 0 for same session, 1 for session change and 2 for new user session
    state_refresh_flags=[]
    outputs=[]

    current_session_indexes=[0 for _ in range(batch_size)]
    current_user_indexes=np.arange(batch_size).tolist()
    next_user_index=batch_size
    start_index=[0 for _ in range(batch_size)]

    end_of_data=False

    i=0
    # user index with the length of alldata means there is no more data
    while not end_of_data:
        input_batch=[]
        state_refresh=[]
        output_batch=[]
        for j in range(batch_size):
            if start_index[j]+len(alldata[current_user_indexes[j]][current_session_indexes[j]])<i+2:
                # End of sessions(user)
                if current_session_indexes[j]+1 == len(alldata[current_user_indexes[j]]):
                    if next_user_index > len(alldata)-1:
                        end_of_data=True
                        break
                    else:
                        current_user_indexes[j]=next_user_index
                        current_session_indexes[j]=0
                        start_index[j]=i
                        next_user_index+=1
                        state_refresh.append(2)
                else:
                    current_session_indexes[j]+=1
                    start_index[j]=i
                    state_refresh.append(1)
            else:
                state_refresh.append(0)
            input_batch.append(alldata[current_user_indexes[j]][current_session_indexes[j]][i-start_index[j]])
            output_batch.append(alldata[current_user_indexes[j]][current_session_indexes[j]][i-start_index[j]+1])
        if end_of_data:
            break

        inputs.append(input_batch)
        outputs.append(output_batch)
        state_refresh_flags.append(state_refresh)
        i+=1
    assert len(inputs)==len(outputs)
    assert len(outputs)==len(state_refresh_flags)

    return inputs,outputs,state_refresh_flags,len(item_dict)

if __name__=='__main__':
    # test.csv => test.json
    input_data,output_data,state_data,item_len=make_train_eval_batch_data('test',3)
    model=HGRU4Rec(s_size=4,u_size=3,batch_size=3,s_dropout=0.5,u_dropout=0.5,len_items=item_len,s_depth=3,u_depth=2)
    model.train(input_data,output_data,state_data,50,'test')