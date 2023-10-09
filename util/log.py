import os
import sys
import os.path as osp
from collections import OrderedDict
import tensorflow as tf

class Text:
  def __init__(self, filename_or_file):
    if isinstance(filename_or_file, str):
      self.file = open(filename_or_file, 'wt')
      self.own_file = True
    else:
      assert hasattr(filename_or_file, 'read'), 'expected file or str, got %s'%filename_or_file
      self.file = filename_or_file
      self.own_file = False
  def writekvs(self, kvs, i, t):
    raw_keys = kvs.keys()
    raw_vals = kvs.values()
    suffix = []
    keys = []
    vals = []

    key2str = OrderedDict()
    for k, v in zip(raw_keys, raw_vals):
      ksp = k.split('/')
      suffix.append(ksp[0])
      keys.append(ksp[-1])
      vals.append(f'{v:<10.4g}')
    keyw = max(map(len, keys))
    valw = max(map(len, vals))
    totalw = keyw + valw + 4
    lw = totalw // 2
    rw = totalw //2 + totalw % 2

    t_txt = f'[ Iteration {i} ]'
    b_txt = f' {t:.3f} '

    top_n = len(t_txt)
    bot_n = len(b_txt)

    top1 = top_n // 2
    top2 = (top_n + 1) % 2
    bot1 = bot_n // 2
    bot2 = (bot_n + 1) % 2

    tr = '╔'+'═'*(lw-top1)+t_txt+'═'*(rw-top1+top2)+'╗'
    mr = '╟'+'─'*(keyw+2) + '┼' +'─'*(valw+2)      +'╢'
    br = '╚'+'═'*(lw-bot1)+b_txt+'═'*(rw-bot1+bot2)+'╝'

    lines = [tr]
    b = ' '
    prev_sf = suffix[0]
    for sf, k, v in zip(suffix, keys, vals):
      if sf != prev_sf:
        lines.append(mr)
      prev_sf = sf
      lines.append(f'║ {k}{b*(keyw - len(k))} │ {v}{b*(valw - len(v))} ║')
    lines.append(br)
    self.file.write('\n'.join(lines) + '\n')
    self.file.flush()

  def writeseq(self, seq):
    seq = list(seq)
    for (i, elem) in enumerate(seq):
      self.file.write(elem)
      if i < len(seq) - 1: # add space unless this is the last one
        self.file.write(' ')
    self.file.write('\n')
    self.file.flush()

  def close(self):
    if self.own_file:
      self.file.close()

class CSV:
  def __init__(self, filename):
    self.file = open(filename, 'w+t')
    self.t = 0
    self.keys = []
    self.sep = ','
  def writekvs(self, kvs, i, t):
    kvs = dict(zip([k.split('/')[-1] for k in kvs.keys()], kvs.values()))
    extra_keys = list(kvs.keys() - self.keys)
    if extra_keys:
      self.keys.extend(extra_keys)
      self.file.seek(0)
      lines = self.file.readlines()
      self.file.seek(0)
      for (i, k) in enumerate(self.keys):
        if i > 0:
          self.file.write(',')
        self.file.write(k)
      self.file.write('\n')
      for line in lines[1:]:
        self.file.write(line[:-1])
        self.file.write(',' * len(extra_keys))
        self.file.write('\n')
    for (i, k) in enumerate(self.keys):
      if i > 0:
        self.file.write(',')
      v = kvs.get(k)
      if v is not None:
        self.file.write(str(float(v)))
    self.file.write('\n')
    self.file.flush()
  def close(self):
    self.file.close()

class TensorBoard:
  def __init__(self, tb_dir):
    os.makedirs(tb_dir, exist_ok=True)
    path = osp.abspath(tb_dir)
    self.writer = tf.summary.create_file_writer(path)
  def writekvs(self, kvs, i, t):
    with self.writer.as_default():
      for k, v in kvs.items():
        tf.summary.scalar(k, v, step=i)
    self.writer.flush()
  def close(self):
    if self.writer:
      self.writer.close()
      del self.writer

def make_output(format_str, ev_dir):
  os.makedirs(ev_dir, exist_ok=True)
  if format_str == 'stdout':
    return Text(sys.stdout)
  elif format_str == 'csv':
    return CSV(osp.join(ev_dir, 'progress.csv'))
  elif format_str == 'tb':
    return TensorBoard(osp.join(ev_dir, 'tb'))

class Logger(object):
  def __init__(self, dir, output_formats):
    self.name2val = {}
    self.dir = dir
    self.output_formats = output_formats

  def logkv(self, key, val):
    self.name2val[key] = val

  def dumpkvs(self, i, t):
    d = self.name2val
    od = OrderedDict([(k, d[k]) for k in sorted(d.keys())])
    for fmt in self.output_formats:
      fmt.writekvs(od, i, t)
    d.clear()

  def log(self, *args):
    for fmt in self.output_formats:
      if isinstance(fmt, Text):
        fmt.writeseq(map(str, args))

  def close(self):
    for fmt in self.output_formats:
      fmt.close()

def record_tab(key, val):
  if logger:
    logger.logkv(key, val)

def dump_tab(i, t):
  if logger:
    logger.dumpkvs(i, t)

def log(*args):
  if logger:
    logger.log(*args)

def close():
  if logger:
    logger.close()

logger = None

def configure(path, formats):
  global logger
  assert type(path) is str and type(formats) is list
  os.makedirs(path, exist_ok=True)
  log_suffix = ''
  output_formats = [make_output(f, path) for f in formats]
  logger = Logger(dir=path, output_formats=output_formats)
  log('Logging to %s' % path)
  return path
