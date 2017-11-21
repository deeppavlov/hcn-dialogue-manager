"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sqlite3


class DatabaseSimulator(object):

    def __init__(self, fname):
        print("database path",fname)
        self.conn = sqlite3.connect(fname)
        self.cursor = self.conn.cursor()
        self.fields = []

    def create_table(self, fields, types, tname='restaurants'):
        self.fields = fields
        f_types = ('{} {} primary key'.format(f, t) if f == 'R_name'
                   else '{} {}'.format(f, t)
                   for f, t in zip(fields, types))
        self.cursor.execute('CREATE TABLE IF NOT EXISTS {} ({})'
                            .format(tname, ', '.join(f_types)))

    def check_if_table_exists(self, tname='restaurants'):
        self.cursor.execute("SELECT name FROM sqlite_master"
                            " WHERE type='table'"
                            " AND name='{}';".format(tname))
        return bool(self.cursor.fetchall())

    def _check_if_resto_exists(self, resto, tname='restaurants'):
        logic_expr = ' AND '.join(['{}=\'{}\''.format(f, v)
                                   for f, v in resto.items()])
        return bool(self.cursor.execute("SELECT EXISTS("
                                        " SELECT 1 FROM {}"
                                        " WHERE {})".format(tname, logic_expr))
                    .fetchone()[0])

    def get_resto_info(self, name, tname='restaurants'):
        if not self.fields:
            self.fields = self.get_field_names()
        ffields = ', '.join(self.fields) or '*'
        fetched = self.cursor.execute("SELECT {} FROM {}"
                                      " WHERE R_name='{}'"
                                      .format(ffields, tname, name))\
            .fetchone() or ()
        return {f: v for f, v in zip(self.fields, fetched)}

    def _update_one(self, name, info, tname='restaurants'):
        set_expr = ', '.join(["{} = '{}'".format(f, v)
                              for f, v in info.items() if f != 'R_name'])
        where_expr = "R_name = '{}'".format(name)
        self.cursor.execute("UPDATE {}"
                            " SET {}"
                            " WHERE {};".format(tname, set_expr, where_expr))

    def insert_one(self, resto, tname='restaurants'):
        if not resto:
            print("DatabaseSimulator error: empty restaurant properties.")
            return
        if not self.fields:
            fields = list(resto.keys())
            types = ('integer' if type(resto[f]) == int else 'text'
                     for f in fields)
            self.create_table(fields, types)

        fformat = '(' + ','.join(['?']*len(self.fields)) + ')'
        db_resto = self.get_resto_info(resto['R_name'])
        if db_resto:
            if db_resto != resto:
                self._update_one(resto['R_name'], resto)
        else:
            self.cursor.execute("INSERT into {} VALUES {}"
                                .format(tname, fformat),
                                [resto.get(f, 'UNK') for f in self.fields])
        self.conn.commit()

    def insert_many(self, restos, tname='restaurants'):
        if not restos or type(restos) is not list:
            print("DatabaseSimulator error: wrong restaurants format")
            return
        if not self.fields:
            fields = list(restos[0].keys())
            types = ('integer' if type(restos[0][f]) == int else 'text'
                     for f in fields)
            self.create_table(fields, types)

        fformat = '(' + ','.join(['?']*len(self.fields)) + ')'
        r_to_insert = []
        for r in restos:
            db_resto = self.get_resto_info(r['R_name'])
            if db_resto:
                if db_resto != r:
                    self._update_one(r['R_name'], r)
            else:
                r_to_insert.append(r)
        if r_to_insert:
            self.cursor.executemany("INSERT into {} VALUES {}"
                                    .format(tname, fformat),
                                    [[r.get(f, 'UNK') for f in self.fields]
                                     for r in r_to_insert])
        self.conn.commit()

    def get_field_names(self, tname='restaurants'):
        self.cursor.execute('PRAGMA table_info({});'.format(tname))
        return [info[1] for info in self.cursor]

    def get_field_types(self, tname='restaurants'):
        self.cursor.execute('PRAGMA table_info({});'.format(tname))
        return [info[2] for info in self.cursor]

    def wrap_selection(self, selection):
        if not self.fields:
            self.fields = self.get_field_names()
        return {f: v for f, v in zip(self.fields, selection)}

    def search(self, properties=None, order_by=None, ascending=False,
               tname='restaurants'):
        order = 'ASC' if ascending else 'DESC'
        if not self.fields and not self.check_if_table_exists():
            return []
        if not properties:
            # get all table content
            if order_by is not None:
                self.cursor.execute("SELECT * FROM {}"
                                    " ORDER BY {} {}"
                                    .format(tname, order_by, order))
            else:
                self.cursor.execute('SELECT * FROM {}'.format(tname))
        else:
            keys = list(properties.keys())
            where_expr = ' AND '.join(['{}=?'.format(k) for k in keys])
            if order_by is not None:
                self.cursor.execute("SELECT * FROM {}"
                                    " WHERE {}"
                                    " ORDER BY {} {}"
                                    .format(tname, where_expr, order_by,
                                            order),
                                    [properties[k] for k in keys])
            else:
                self.cursor.execute("SELECT * FROM {}"
                                    " WHERE {}".format(tname, where_expr),
                                    [properties[k] for k in keys])
        return list(map(self.wrap_selection, self.cursor.fetchall() or []))
